"""量子化変種の品質評価に使うプロンプト集。

設計の要点は 2 つ。

1. **鋭い分布の課題を混ぜる。** 自由記述の指示文だけだと次トークン分布が広く、
   量子化の損傷が KLD のノイズに埋もれる。逐語コピー、パターン継続、コードの
   続き、筆算のように「正解が 1 つに決まる」課題を入れると、わずかな損傷が
   top-1 の反転として出る。

2. **どの部品を叩くかを prompt ごとに書く。** Flash-Next は experts (MoE) と
   n-gram ハッシュ表という、性格の違う大きな塊にビットを配る。どちらに盛るか
   を決めるには、指標を 1 つの平均値に潰さず、部品別に読めるようにする必要が
   ある。`stress` がその札。

   ngram    局所の字面。逐語コピー、パターン反復、同じ識別子の再出現、
            珍しい表記。PLE の n-gram 表 (51B params) が効く領域
   experts  領域知識。事実想起、専門分野、言語ごとの語彙。MoE 本体
   attn     離れた位置の参照。長文からの値の拾い出し
   struct   構文の型。JSON/YAML/表のような閉じた形式
   reason   多段の推論

`why` は「その項目で何を見るか」。結果を読むときに、数字だけ見て意味を後付け
しないための備忘。
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EvalPrompt:
    key: str
    text: str
    stress: tuple[str, ...]
    lang: str
    why: str


# 長文リコール用の資料。数値をばらけさせて、拾い出しに位置の特定が要るようにする
_LONGDOC = """以下は社内の在庫棚卸し記録である。

2026-03-04 第一倉庫 A 棚: 型番 RX-4410 が 187 個、型番 RX-4412 が 42 個。担当は棚橋。
2026-03-04 第一倉庫 B 棚: 型番 TZ-9003 が 1,204 個、型番 TZ-9007 が 96 個。担当は棚橋。
2026-03-05 第二倉庫 A 棚: 型番 RX-4410 が 63 個、型番 QM-118 が 8 個。担当は苅田。
2026-03-05 第二倉庫 C 棚: 型番 TZ-9003 が 512 個、型番 QM-118 が 3,340 個。担当は苅田。
2026-03-06 第三倉庫 A 棚: 型番 RX-4412 が 771 個、型番 TZ-9007 が 1,905 個。担当は的場。
2026-03-06 第三倉庫 B 棚: 型番 QM-118 が 27 個、型番 RX-4410 が 4,096 個。担当は的場。
2026-03-09 第一倉庫 C 棚: 型番 TZ-9007 が 350 個、型番 QM-118 が 61 個。担当は棚橋。
2026-03-09 第二倉庫 B 棚: 型番 RX-4412 が 2,048 個、型番 TZ-9003 が 15 個。担当は苅田。

なお 2026-03-05 の第二倉庫の記録は、担当者の申告により 3 月 11 日に再計測され、
QM-118 のみ 3,340 個から 3,298 個へ訂正された。他の型番に変更はない。"""

_CODE_HEAD = '''def rolling_window(seq, size, *, step=1, drop_last=True):
    """連続する部分列を size 個ずつ切り出す。

    step だけずらしながら進み、drop_last が真なら長さが足りない末尾を捨てる。
    """
    if size <= 0:
        raise ValueError("size は 1 以上")
    if step <= 0:
        raise ValueError("step は 1 以上")
    out = []
    i = 0
    while i < len(seq):'''


_ITEMS: tuple[EvalPrompt, ...] = (
    # ---------------------------------------------------------------- 逐語
    EvalPrompt(
        key="copy-rare-ja",
        text=(
            "次の文字列を一字一句そのまま出力してください。説明や補足は不要です。\n\n"
            "檜垣ノ辻三丁目七番地弐拾五号 藤原朝臣惟仲 令和七年拾弐月参拾壱日 "
            "受付番号 甲第肆佰玖拾参號 担当 苅田薫子"
        ),
        stress=("ngram",),
        lang="ja",
        why="珍しい表記の逐語コピー。直前 2 トークンから次が一意に決まるので、"
        "n-gram 表が潰れると真っ先に崩れる",
    ),
    EvalPrompt(
        key="copy-hex",
        text=(
            "次の識別子を順にそのまま書き写してください。並び順も変えないこと。\n\n"
            "3f9a2c81-77de-4b05-9e13-a0cc85f21b6d\n"
            "0e4d17bb-2f80-49ca-8d77-5b1ae9c3042f\n"
            "c81b60a4-9d3e-4f12-b7a8-2e6f0d59173c\n"
            "a72f38d0-1c95-4e6b-90af-33b8e7d24610"
        ),
        stress=("ngram",),
        lang="en",
        why="語彙の裾にある短い断片の連続。埋め込みと n-gram の両方に効く。"
        "自由度がほぼ無いので損傷がそのまま誤りになる",
    ),
    EvalPrompt(
        key="pattern-table",
        text=(
            "次の表を、同じ規則のまま id が 0021 になるまで続けてください。"
            "書式は一切変えないこと。\n\n"
            "| id | code | qty |\n"
            "|------|--------|-----|\n"
            "| 0011 | AX-011 |  33 |\n"
            "| 0012 | AX-012 |  36 |\n"
            "| 0013 | AX-013 |  39 |"
        ),
        stress=("ngram", "struct"),
        lang="ja",
        why="厳密な反復。行の骨格は n-gram、数の増分は推論。両方が要る",
    ),
    EvalPrompt(
        key="edit-preserve",
        text=(
            "次の関数に型ヒントと docstring を追加してください。"
            "それ以外は 1 文字も変えないでください。\n"
            "```python\n"
            "def merge(a, b):\n"
            "    out = dict(a)\n"
            "    for k, v in b.items():\n"
            "        if k in out and isinstance(out[k], dict) and isinstance(v, dict):\n"
            "            out[k] = merge(out[k], v)\n"
            "        else:\n"
            "            out[k] = v\n"
            "    return out\n"
            "```"
        ),
        stress=("ngram", "code"),
        lang="ja",
        why="大半が入力の写しになる編集課題。実務で最も体感差が出る形",
    ),
    # ---------------------------------------------------------------- コード
    EvalPrompt(
        key="code-continue",
        text=(
            "次の関数の続きを書いてください。既存の行は変えず、続きだけを出力すること。\n"
            "```python\n" + _CODE_HEAD + "\n```"
        ),
        stress=("code", "ngram"),
        lang="ja",
        why="変数名が既出。識別子の再出現は n-gram の得意領域で、"
        "構文の型は experts 側。続きが一意に近いので分布が鋭い",
    ),
    EvalPrompt(
        key="code-py",
        text=(
            "Python で、ディレクトリ以下の全ファイルを SHA-256 でハッシュ化して"
            "重複ファイルを検出するスクリプトを書いてください。"
        ),
        stress=("code", "experts"),
        lang="ja",
        why="標準ライブラリの知識。生成の自由度は高いが語彙は締まっている",
    ),
    EvalPrompt(
        key="code-rust",
        text=(
            "Write a Rust function that parses an ISO-8601 timestamp without external "
            "crates, returning a struct with year, month, day, hour, minute, second."
        ),
        stress=("code", "experts"),
        lang="en",
        why="Python 以外の構文。言語ごとに違う expert が点くはずで、"
        "ルーティングの広がりを稼ぐ",
    ),
    EvalPrompt(
        key="code-debug",
        text=(
            "次の関数にはバグがあります。原因を指摘し、最小の修正を示してください。\n"
            "```python\n"
            "def chunk(seq, n):\n"
            "    out = []\n"
            "    for i in range(0, len(seq), n):\n"
            "        out.append(seq[i:i + n])\n"
            "    if len(out[-1]) < n:\n"
            "        out.pop()\n"
            "    return out\n"
            "```\n"
            "なお、空の入力でも落ちないことと、末尾の端数を捨てないことが要件です。"
        ),
        stress=("code", "reason"),
        lang="ja",
        why="指摘すべき箇所が決まっているので分布が鋭い。"
        "コード読解が量子化でどれだけ鈍るかが出る",
    ),
    EvalPrompt(
        key="code-refactor",
        text=(
            "次のコードを、振る舞いを変えずに読みやすくしてください。"
            "関数名と引数名は変えないでください。\n"
            "```python\n"
            "def f(d):\n"
            "    r = {}\n"
            "    for k in d:\n"
            "        if type(d[k]) == dict:\n"
            "            for k2 in d[k]:\n"
            "                r[k + '.' + k2] = d[k][k2]\n"
            "        else:\n"
            "            r[k] = d[k]\n"
            "    return r\n"
            "```"
        ),
        stress=("code", "ngram"),
        lang="ja",
        why="入力の写しが多い書き換え。edit-preserve のコード版",
    ),
    EvalPrompt(
        key="code-ts",
        text=(
            "TypeScript で、ネストしたオブジェクトのキーをドット区切りの文字列型として "
            "取り出す型 `DeepKeys<T>` を書いてください。配列は葉として扱ってください。"
        ),
        stress=("code", "experts"),
        lang="ja",
        why="型レベルの再帰。Python/Rust とはまた違う語彙帯で expert の広がりを稼ぐ",
    ),
    EvalPrompt(
        key="code-go",
        text=(
            "Write a Go function that fans out work to N goroutines, collects results "
            "through a channel, and cancels the remaining work as soon as any worker "
            "returns an error. Use context.Context."
        ),
        stress=("code", "experts"),
        lang="en",
        why="並行処理の定型。イディオムが強く決まっている領域",
    ),
    EvalPrompt(
        key="code-c",
        text=(
            "C言語で、任意長の行を読み込む関数 `char *read_line(FILE *fp)` を"
            "書いてください。realloc で伸ばし、EOF と読み込み失敗を区別できるように"
            "してください。"
        ),
        stress=("code", "experts"),
        lang="ja",
        why="手動のメモリ管理。高水準言語と別の語彙と定型",
    ),
    EvalPrompt(
        key="code-shell",
        text=(
            "Apache のアクセスログ (combined 形式) から、直近のステータス 5xx を"
            "URL ごとに集計し、件数の多い順に上位 10 件を出す awk のワンライナーを"
            "書いてください。"
        ),
        stress=("code", "ngram"),
        lang="ja",
        why="記号の密度が高く、1 文字の違いが致命的になる。字面の精度が出る",
    ),
    EvalPrompt(
        key="code-algo",
        text=(
            "重み付き有向グラフの単一始点最短経路を、優先度付きキューを使った"
            "ダイクストラ法で実装してください。Python、標準ライブラリのみ。"
            "到達不能な頂点は無限大として返してください。"
        ),
        stress=("code", "reason"),
        lang="ja",
        why="定番アルゴリズム。手順が固定なので損傷が構造の崩れとして出る",
    ),
    EvalPrompt(
        key="code-sql",
        text=(
            "PostgreSQL で、注文テーブル orders(id, customer_id, placed_at, total) と"
            "顧客テーブル customers(id, name, region) から、"
            "地域ごとに直近 90 日の売上上位 3 顧客を求めるクエリを書いてください。"
            "ウィンドウ関数を使うこと。"
        ),
        stress=("code", "experts", "struct"),
        lang="ja",
        why="宣言的な構文と予約語の連なり。汎用コードとは別の語彙帯",
    ),
    # ---------------------------------------------------------------- 事実
    EvalPrompt(
        key="ja-fact",
        text="鎌倉幕府の成立から滅亡までの主要な出来事を、年号付きで時系列に列挙してください。",
        stress=("experts",),
        lang="ja",
        why="固有名詞と年号。日本史の知識が載っている expert を叩く",
    ),
    EvalPrompt(
        key="en-fact",
        text=(
            "List the chemical elements discovered in the 20th century, with the year "
            "and discoverer for each."
        ),
        stress=("experts",),
        lang="en",
        why="別領域の固有名詞。ja-fact と違う expert が点くことを期待する",
    ),
    EvalPrompt(
        key="fact-med",
        text=(
            "抗凝固薬のワルファリンとダビガトランについて、作用機序、"
            "モニタリングの要否、主な相互作用の違いを説明してください。"
        ),
        stress=("experts", "ngram"),
        lang="ja",
        why="医薬品名は語彙の裾。専門領域の expert と珍しい表記の両方に効く",
    ),
    # ---------------------------------------------------------------- 推論
    EvalPrompt(
        key="reason-logic",
        text=(
            "5 人が横一列に並んでいる。A は B の右隣ではない。C は左端でも右端でもない。"
            "D は A より左にいる。E は B の隣にいる。C は D の 2 つ右にいる。"
            "並び順として成立するものをすべて挙げ、そこに至る過程も示してください。"
        ),
        stress=("reason", "experts"),
        lang="ja",
        why="制約充足。途中で分岐するので、わずかな損傷が経路の選択を変えやすい",
    ),
    EvalPrompt(
        key="math-arith",
        text=(
            "次の計算を筆算の手順を示しながら行ってください。\n"
            "(1) 48273 × 6109\n"
            "(2) 9182736 ÷ 47 の商と余り\n"
            "(3) 2^31 - 2^17 を 10 進で"
        ),
        stress=("reason",),
        lang="ja",
        why="桁ごとに正解が一意。数字トークンは分布が鋭いので損傷が可視化される",
    ),
    EvalPrompt(
        key="math-count",
        text="3 桁の整数のうち、各桁の数字の和が 10 になるものは何個あるか。途中の考え方も含めて答えてください。",
        stress=("reason",),
        lang="ja",
        why="数え上げ。手順の型は決まっているが分岐がある",
    ),
    # ---------------------------------------------------------------- 構造
    EvalPrompt(
        key="json-struct",
        text=(
            "架空の書店の在庫管理 API のレスポンス例を JSON で作成してください。"
            "書籍 5 冊分、各書籍には ISBN、タイトル、著者、価格、在庫数を含めてください。"
        ),
        stress=("struct", "ngram"),
        lang="ja",
        why="括弧とキーの反復。構文の骨格は局所的で n-gram に載りやすい",
    ),
    EvalPrompt(
        key="yaml-config",
        text=(
            "Kubernetes の Deployment マニフェストを YAML で書いてください。"
            "レプリカ 3、コンテナは nginx:1.27、リソース要求は CPU 250m / メモリ 256Mi、"
            "liveness と readiness の両方のプローブを含めること。"
        ),
        stress=("struct", "experts"),
        lang="ja",
        why="インデントが意味を持つ形式。字下げの誤りが即座に出る",
    ),
    # ---------------------------------------------------------------- 多言語
    EvalPrompt(
        key="zh",
        text="请用中文解释一下什么是投机解码（speculative decoding），以及它为什么能加速大语言模型的推理。",
        stress=("experts", "ngram"),
        lang="zh",
        why="別の文字体系。中国語の語彙と expert が日本語と分かれているかを見る",
    ),
    EvalPrompt(
        key="ko",
        text="한국어로 트랜스포머의 어텐션 메커니즘이 왜 순환 신경망보다 병렬화에 유리한지 설명해 주세요.",
        stress=("experts", "ngram"),
        lang="ko",
        why="学習量が相対的に少ない言語。量子化の損傷が最初に出る帯",
    ),
    EvalPrompt(
        key="translate",
        text=(
            "次の文を自然な英語に翻訳してください:"
            "「量子化は精度と引き換えにメモリと帯域を節約する技術であり、"
            "その配分には測定に基づく判断が必要である。」"
        ),
        stress=("experts", "ngram"),
        lang="ja",
        why="入力に強く縛られた生成。訳語選択が分布の鋭い分岐になる",
    ),
    # ---------------------------------------------------------------- 長文
    EvalPrompt(
        key="longctx-recall",
        text=(
            _LONGDOC + "\n\n"
            "上の記録から、型番 QM-118 の総数を求めてください。"
            "訂正がある場合は訂正後の値を使い、どの行を足したかも示してください。"
        ),
        stress=("attn", "reason"),
        lang="ja",
        why="離れた位置の値の拾い出しと、末尾の訂正の適用。"
        "attention 側の損傷を見るための項目で、n-gram では拾えない",
    ),
    EvalPrompt(
        key="longctx-quote",
        text=(_LONGDOC + "\n\n上の記録のうち、担当が的場である行をそのまま抜き出してください。"),
        stress=("attn", "ngram"),
        lang="ja",
        why="離れた位置からの逐語抜き出し。attention で位置を決め、"
        "字面は n-gram が支える。2 つの部品の合わせ技",
    ),
    # ---------------------------------------------------------------- 自由記述
    EvalPrompt(
        key="ja-explain",
        text="分散システムにおける結果整合性と強整合性の違いを、具体例を挙げながら詳しく説明してください。",
        stress=("experts",),
        lang="ja",
        why="自由記述の基準線。分布が広いので、鋭い項目との差を読むための対照",
    ),
    EvalPrompt(
        key="en-prose",
        text=(
            "Explain why the sky is blue during the day but red at sunset, in a way a "
            "curious teenager would enjoy."
        ),
        stress=("experts",),
        lang="en",
        why="英語の自由記述。ja-explain と対にして言語差を見る",
    ),
    EvalPrompt(
        key="summarize",
        text=(
            "次の主張を 3 行で要約してください: 大規模言語モデルの推論速度はメモリ帯域に"
            "律速されることが多い。重みを低ビットに量子化すると読み出し量が減って速度が"
            "上がるが、精度が犠牲になる。投機デコードは複数トークンを一括検証することで、"
            "帯域あたりの生成トークン数を増やす。この 2 つは独立に効くため併用できるが、"
            "量子化はドラフトの受理率を下げる方向にも働くため、"
            "併用時の利得は単純な掛け算にはならない。"
        ),
        stress=("attn", "experts"),
        lang="ja",
        why="入力の言い換え。原文への依存が強く、写しと生成の中間",
    ),
)

PROMPTS: dict[str, EvalPrompt] = {p.key: p for p in _ITEMS}

STRESS_KINDS: tuple[str, ...] = ("ngram", "experts", "attn", "struct", "reason", "code")


def keys_by_stress(kind: str) -> list[str]:
    """その部品を叩く prompt の key を返す。"""

    return [k for k, p in PROMPTS.items() if kind in p.stress]


def summary() -> str:
    lines = [f"{len(PROMPTS)} prompts"]
    for kind in STRESS_KINDS:
        ks = keys_by_stress(kind)
        lines.append(f"  {kind:8s} {len(ks):2d}  {', '.join(ks)}")
    langs: dict[str, int] = {}
    for p in PROMPTS.values():
        langs[p.lang] = langs.get(p.lang, 0) + 1
    lines.append("  言語: " + ", ".join(f"{k}={v}" for k, v in sorted(langs.items())))
    return "\n".join(lines)


if __name__ == "__main__":
    print(summary())
