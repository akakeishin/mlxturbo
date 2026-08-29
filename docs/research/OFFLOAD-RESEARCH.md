# SSD offload 調査の結論（2026-08-26。全主張に一次ソースあり、詳細はセッションログ）

## 判定

- **GLM-5.2 (316GB) の SSD offload は割に合わない。** 予測 0.6〜1.1 tok/s
  （中央値 0.8）。独立検証: DeepSeek-R1 671B (212GB) を RAM 96GB + NVMe の
  llama.cpp で 1.0〜1.3 tok/s。同じ桁に着地する。「毎日使える最速」の
  一次目標から外れるため着手しない
- **投機×offload の「expert 和集合を 1 回読んで償却」仮説は先行研究に否定された。**
  和集合は m にほぼ線形で伸び（DraftExpert 実測: verify 1 パス = target step の
  2.35〜3.10 倍）、I/O が黒字になる必要受理率 58〜79% はうちの実測 40〜60% の上。
  M3 Max 実機の flash-moe は draft 併用で正味 4.5 倍の低速化を記録。
  唯一の利得はキュー深度上昇 (1.1〜1.3x) で、膨張の罰と相殺し正味 ~1.0x
- **DeepSeek-V4-Flash (115GB) は offload 案件ではない。RAM に載る。**
  routed expert 総量 104GB + 常駐 10.7GB。iogpu.wired_limit_mb を ~120GB へ
  上げれば常駐でき、帯域律速で 25〜45 tok/s の見込み。しかも
  num_nextn_predict_layers=1 = MTP ヘッド同梱で、fastmlx の L2 が直接乗る。
  やるならこちら（Phase M5）

## 将来 I/O を書くときの macOS 実測知見（flash-moe の M3 Max ログより）

- mmap 禁止（16KB ページで 1 expert = 240 page fault、5 倍遅い。
  community の mmap→Metal 実装は常駐比 1/240 の 0.025 tok/s）。
  MLX 本体も pread 32MB 単位の ThreadPool
- cold read の実力は並列 pread 4 スレッドで 5.5GB/s（warm page cache は
  32GB/s で、「SSD 17.5GB/s」系の数字は warm の誤読）
- 自前 cache を作ると負ける（Metal buffer は wired で page cache を圧迫、
  実測 38% 低速化）。OS の page cache に任せるのが最速
- カーネルヒント (F_RDAHEAD/F_RDADVISE/madvise 系) は全部無効か有害
- Python で I/O スレッドを回すなら GIL を離す設計必須
  （pread 単体 18.8GB/s が MLX と組むと 2.6GB/s に落ちた記録）
- GPU 演算中の先読みは DMA が unified memory を食い GPU 待ち +73% で正味ゼロ

## 主要ソース

LLM in a flash (2312.11514) / Mixtral-offloading (2312.17238) /
PowerInfer-2 (2406.06282) / Not All Models Suit Expert Offloading (2505.16056) /
DraftExpert (2607.24434) / EcoSpec (2607.12696) / SpecOffload (2505.10259) /
flash-moe (github.com/danveloper/flash-moe, M3 Max 実測) /
DeepSeek-R1 671B on 96GB (unsloth discussion)

## 追記: FreeToken (arXiv:2608.16157) の検討

GLM-5.2 753B を 14.9 tok/s で動かした報告だが、実体は SSD offload ではなく
DRAM offload。expert 全量は 512GB のシステム DRAM (178GB/s) に常駐し、SSD は
初回ロードのみ。DRAM に収まらない場合は pure-CPU 落ちで SSD サービング経路は
無い。核心の適応分割 (q*=m·BP/BH) は PCIe と host の 2 帯域ドメインの裁定で、
unified memory (1 ドメイン) には適用不能。よって 128GB Mac での判定は不変。
含意はむしろ逆方向で、M3 Ultra 512GB なら GLM-5.2 が unified memory に常駐し
(帯域は彼らの DDR5 比 4.5 倍)、FreeToken 機を大差で超える理論位置にある。
router 分割・弾性 expert キャッシュ等の実行構造は将来の別バックエンド設計の
参考になる (llama.cpp 比 2 倍の実証)。
