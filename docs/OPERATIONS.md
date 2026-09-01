# OPERATIONS — 公開インスタンスの運用

`mlxturbo-serve` を LAN/WAN に公開するか、あるいは自分専用でも常駐させて他のマシンから叩く場合の
運用メモ。起動オプションや接続方法そのものは [`SERVER.md`](SERVER.md) を参照。ここでは「公開して
放置しても大丈夫にする」ための設定と、動いているかどうかの見方をまとめる。

前提として、このサーバーは 1 プロセスにつき 1 モデルしか持たず、リクエストは既定 (`--max-batch` と
`--max-batch-spec` を省略時) では内部で直列処理する。まとめるフラグは対象ごとに 2 つあり、`--max-batch N`
は非投機 (`FallbackRunner`) の要求、`--max-batch-spec N` は Flash-Next + MTP 投機 (`FlashSpecRunner`) の
要求を対象にする。後者には長さとサンプリングの条件があり、外れた要求は直列のまま。公開インスタンスとして
複数人・複数エージェントに使わせる場合、この制約が真っ先に効いてくる。詳細は
[`SERVER.md`](SERVER.md) の「制約」節。

## リバースプロキシ (nginx / Caddy)

`mlxturbo-serve` 自体は TLS を話さない。`--host 0.0.0.0` で待受を外に開ける場合は、必ず手前に
リバースプロキシを置いて TLS を終端し、以下の 2 点を必ず設定する。

### タイムアウト

このサーバーは長いプレフィルの間ストリーミング接続に SSE keepalive コメント (`: keepalive`) を
15 秒おきに送るが (`SERVER.md` の「SSE keepalive」節)、プロキシ側のタイムアウトがこれより短いと
コネクションを切られる。プレフィルは実測でモデル・プロンプト長次第で数分かかることがある
(97k トークンのプロンプトで約 3 分の実例あり)。プロキシのタイムアウトは keepalive 間隔 (15 秒) より
十分長く、かつ想定する最長プレフィル時間より長く取る。

nginx の例 (`/etc/nginx/conf.d/mlxturbo.conf`):

```nginx
server {
    listen 443 ssl;
    server_name mlxturbo.example.com;

    ssl_certificate     /etc/letsencrypt/live/mlxturbo.example.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/mlxturbo.example.com/privkey.pem;

    location / {
        proxy_pass http://127.0.0.1:8765;
        proxy_http_version 1.1;

        # ストリーミング応答をバッファせずそのまま流す (SSE keepalive を含む)
        proxy_buffering off;
        proxy_cache off;

        # プレフィルが長いモデル・長い会話履歴を想定し、既定 (60s) より長めに取る
        proxy_read_timeout 600s;
        proxy_send_timeout 600s;

        # WebSocket は使わないが、Connection ヘッダを上書きしないこと
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

Caddy の例 (`Caddyfile`):

```
mlxturbo.example.com {
    reverse_proxy 127.0.0.1:8765 {
        flush_interval -1
    }
    # Caddy の読み取りタイムアウトは既定で無制限だが、手前に別の CDN/LB を
    # 挟む場合はそちら側のタイムアウトも同様に長く取ること
}
```

### body サイズ上限

サーバー自身は `--max-context-tokens` でプロンプトのトークン数を制限するが (超過は 400
`context_length_exceeded`)、これはトークン化した後の判定であり、その手前でリバースプロキシが
リクエストボディのバイト数を先に弾ける。長い会話履歴や画像埋め込み込みのリクエストを想定し、
既定の小さい上限 (nginx の既定は 1MB) のままだとトークン数上限に達する前に 413 になる。

nginx: `client_max_body_size 20m;` を `server` または `location` ブロックに追加する。
Caddy: 既定で上限なし。制限したい場合のみ `request_body { max_size 20MB }` を追加する。

サイズは扱うプロンプト長に応じて調整する。目安として、1 トークンあたり UTF-8 で数バイト程度、
数万トークンの会話履歴なら数百KB〜数MBに収まることが多いが、Base64 画像を埋め込む場合はもっと
大きくなる。

## `--api-key`

`--api-key` は認証だけを行い通信は暗号化しない。公開インスタンスでは必ずリバースプロキシで TLS を
終端したうえで使う (`SERVER.md` の「API キー認証」節に詳細)。複数キーを渡してローテーションできる。
鍵そのものはコマンドライン引数として渡すため shell history や `ps` 出力に残り得る点は
`SERVER.md` の注意書きのとおり。共有ホストではプロキシ側の認証 (Basic 認証や mTLS など) と
併用することを検討する。

## `--max-queue`

直列化ロックの待ち行列上限。既定 8。公開インスタンスで想定同時アクセス数が既定より多い場合は
明示的に上げる (ただし上げても直列処理自体は変わらないので、体感の待ち時間が延びるだけ)。上限に
達すると新規リクエストは生成を試みずに即座に 503 (`Retry-After: 1`) を返す。監視側はこの 503 を
「詰まっている」ではなく「意図どおり弾いている」として区別できるようにしておくとよい
(`Retry-After` ヘッダの値どおりリトライすれば通ることが多い)。

## `/health` の監視

`GET /health` は認証の有無に関わらず常に応答する (公開インスタンスでも監視用に鍵なしで叩ける)。
主なフィールドの意味:

`runner` は解決された実行経路。`flash_spec` / `spec` / `fallback` のいずれか。起動時に
`--require-runner` を指定していれば、この値は起動が通った時点で固定されている。指定していない
場合、監視でこの値が想定と違えば「投機が効かないまま動いている」ことを意味する。

`fallback_reason` は `runner` が `fallback` のときだけ現れる。値が無ければ (キー自体が無ければ)
投機経路で動いている。監視ダッシュボードでは「このキーの有無」自体をアラート条件にできる —
本来 `flash_spec` で動くはずのモデルで、再起動後にこのキーが出現したら性能劣化のサインになる。

`queue_depth` は待ち行列 + 処理中の合計。`--max-queue` に近づいている時間が長ければ、公開範囲を
絞るかキュー上限を上げるかの判断材料になる。`busy` (bool) と合わせて見ると、単発の混雑か常時
詰まっているかを区別できる。

`version` はデプロイ確認用。ロードが終わっていない起動直後は 503 とともに `loaded: false` を返す
(この間はプロセスは生きているがモデル未ロード — ヘルスチェックの「起動直後の失敗」と「本当に
落ちた」を混同しないこと)。

監視の組み方の例: `/health` を 15〜30 秒間隔でポーリングし、`status != "ok"` または HTTP ステータス
が 200 以外なら再起動を検討、`fallback_reason` が想定外に出現したら性能低下として別チャンネルに
通知、`queue_depth >= max-queue` が続くならキャパシティ超過として通知、という 3 段構えが最小限。

## ログの読み方

起動ログには一度だけ、バージョン・待ち行列上限・API キー認証の有無・CORS 設定などが出る
(`SERVER.md` 参照)。生成のたびに出る一行ログはこの形式:

```
[mlxturbo-serve] prefill reused=1234 new=56 decode=31.9tok/s
```

`reused` はプロンプトキャッシュ (session ごとの KV) から再利用できたプレフィルのトークン数、
`new` は新規に計算したトークン数。2 ターン目以降の会話で `reused` が毎回 0 に近ければ、
`--max-sessions` の上限に達して会話の KV が捨てられている (LRU で最も長く未使用のものが落ちる)
か、クライアント側が毎回別セッション扱いで送っている可能性を疑う。`decode` はそのリクエストの
decode 速度 (tok/s)。fallback 経路や投機が効いていない状態が続くと、この値は
`--require-runner` 無しで運用している場合に気づきにくい劣化のサインになる — `/health` の
`runner`/`fallback_reason` と合わせて見ること。

## launchd で常駐させる

Mac を再起動しても自動で立ち上がるようにする場合は launchd の plist を使う。設定例は
[`../contrib/launchd/com.mlxturbo.serve.plist`](../contrib/launchd/com.mlxturbo.serve.plist)
に置いてある。ポート・モデルパス・ログ出力先はコメントの指示に従って書き換える。
