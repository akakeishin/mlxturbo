// metal_probe — MLX が出す Metal の dispatch を数え、command buffer の GPU
// 実行時間 (GPUStartTime/GPUEndTime) を集める観測専用の dylib。
//
// なぜ swizzle なのか (先に潰した経路):
//   - xctrace (Metal System Trace) は python (MLX) の Compute 区間をほぼ拾えな
//     かった (docs/research/SESSION-2026-09-02-CATCHUP.md「xctrace は今回は
//     使えなかった」節)。
//   - mx.metal.start_capture の .gputrace は 98GB の常駐リソースを丸ごと書き
//     出すので実モデルでは 68GB まで膨らんで終わらない
//     (tools/gpu_capture_forward.py 冒頭の実測記録)。
//   - MTLCounterSampleBuffer を使う筋は MLX の Python API が command buffer /
//     encoder を露出していないので届かない (同上)。
//   ここでは Metal の**具象クラスのメソッドを実行時に差し替える**ことで、
//   MLX の内部に手を入れずに同じ情報 (dispatch 回数・カーネル名・command
//   buffer ごとの GPU 実行区間) を取る。libmlx にはリンクしない。
//
// 差し替える 6 種:
//   -[<MTLComputeCommandEncoder> setComputePipelineState:]      現在のカーネル名
//   -[<MTLComputeCommandEncoder> dispatchThreadgroups:threadsPerThreadgroup:]
//   -[<MTLComputeCommandEncoder> dispatchThreads:threadsPerThreadgroup:]
//   -[<MTLComputeCommandEncoder> dispatchThreadgroupsWithIndirectBuffer:...]
//   -[<MTLCommandBuffer> commit]                completion handler で GPU 時間
//   -[<MTLDevice> newComputePipelineStateWith...]  pipeline → カーネル名の対応
//
// **Metal のデバイスが出来たあと (= MLX が GPU を 1 回でも使ったあと) に
// install すること。**ドライバのクラスはそれまでロードされない。
//
// ## 差し替えの作り (2026-09-03 に SIGSEGV で 1 回踏んだ穴)
//
// 差し替え関数を 1 個にして「受け手のクラス階層をたどって元の IMP を引く」
// 作りにすると**無限再帰で落ちる**。実測のスタック:
//
//     rep_commit → -[AGXG15XFamilyCommandBuffer commit] → rep_commit → ...
//
// 原因は、`commit` を **AGXG15XFamilyCommandBuffer とその親クラスの両方**が
// 自分で実装していて両方差し替わり、AGX の commit が中で `[super commit]` を
// 呼ぶと親側の差し替えに入り、そこから「受け手のクラス」= AGX で元の IMP を
// 引き直すので AGX の commit に戻ってしまうため。
//
// なので **差し替えたクラス 1 個につき差し替え関数を 1 個**用意し
// (`template<int I>` の trampoline を 16 本ずつ)、それぞれが自分が置き換えた
// IMP だけを呼ぶ。`[super ...]` の連鎖はこれで正しく降りる。そのぶん同じ
// 呼び出しが複数の段で観測されるので、**数えるのは最外周だけ** (スレッド
// ローカルの深さカウンタ) にする。
//
// もう 1 つ: `commit` のようにありふれた名前のセレクタは Metal と無関係の
// クラスも持っている。差し替える前に「Metal のそれ」だと分かる別のセレクタも
// 一緒に持っているかを見て絞る (`addCompletedHandler:` / `GPUStartTime` など)。
//
// ビルド: ./build_metal_probe.sh  ->  tools/bridge/libmetal_probe.dylib

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <objc/message.h>
#import <objc/runtime.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <cctype>
#include <cstring>
#include <mutex>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#define MP_EXPORT extern "C" __attribute__((visibility("default")))

namespace {

constexpr int MAXS = 48;  // 1 セレクタあたり差し替えられるクラス数の上限

// ---------------------------------------------------------------------------
// 集計の入れ物
// ---------------------------------------------------------------------------
std::mutex g_mu;
std::atomic<bool> g_on{false};
// mp_debug(n) で、pipeline まわりの「名前の取れそうな場所」を先頭 n 件だけ
// stderr に吐く。どこに MLX のカーネル名が入っているかを確かめるための口。
std::atomic<int> g_dbg_left{0};
std::atomic<long> g_pending{0};  // completion handler 待ちの command buffer 数

std::unordered_map<std::string, int> g_ids;
std::vector<std::string> g_names;
std::vector<uint64_t> g_counts;   // カーネル名ごとの dispatch 回数
std::vector<double> g_gpu_ms;     // カーネル名ごとの GPU 時間 (CB を等分した配分)

uint64_t g_dispatch = 0;
uint64_t g_cb = 0;                // commit した command buffer の数
uint64_t g_cb_with_dispatch = 0;
double g_gpu_sum_ms = 0.0;        // CB の (GPUEnd - GPUStart) の総和
std::vector<std::pair<double, double>> g_ivals;  // CB の GPU 区間 (union 用)

std::unordered_map<const void*, std::string> g_pipe_name;  // pipeline -> カーネル名

// MLX は 1 本のスレッドで順にエンコードするので、「今の pipeline」も
// 「この CB に積んだ dispatch の名前」もスレッドローカルの 1 個で足りる。
thread_local int t_pipe = -1;
thread_local std::vector<int>* t_pending_names = nullptr;

// 最外周だけ数えるための深さカウンタ (上のコメント参照)
thread_local int t_d_dtg = 0;
thread_local int t_d_dth = 0;
thread_local int t_d_dti = 0;
thread_local int t_d_pipe = 0;
thread_local int t_d_commit = 0;

struct Depth {
  int* p;
  bool outer;
  explicit Depth(int* c) : p(c), outer(*c == 0) { *c += 1; }
  ~Depth() { *p -= 1; }
};

int intern_locked(const std::string& s) {
  auto it = g_ids.find(s);
  if (it != g_ids.end()) {
    return it->second;
  }
  int id = static_cast<int>(g_names.size());
  g_ids.emplace(s, id);
  g_names.push_back(s);
  g_counts.push_back(0);
  g_gpu_ms.push_back(0.0);
  return id;
}

void note_dispatch() {
  std::lock_guard<std::mutex> lk(g_mu);
  int id = t_pipe >= 0 ? t_pipe : intern_locked("(pipeline unknown)");
  g_dispatch += 1;
  if (id >= 0 && id < static_cast<int>(g_counts.size())) {
    g_counts[static_cast<size_t>(id)] += 1;
  }
  if (!t_pending_names) {
    t_pending_names = new std::vector<int>();
  }
  t_pending_names->push_back(id);
}

std::string name_for_pipe(id pso) {
  const void* key = static_cast<const void*>(pso);
  {
    std::lock_guard<std::mutex> lk(g_mu);
    auto it = g_pipe_name.find(key);
    if (it != g_pipe_name.end()) {
      return it->second;
    }
  }
  // 台帳に無いとき (作成の経路を掴めていない pipeline)。pipeline state 自身に
  // 名前を持っていそうな引数なしのアクセサを順に当てる。
  // 2026-09-03 の実測: MLX は `newComputePipelineStateWithDescriptor:options:
  // reflection:error:` を通らない (差し替えても 1 回も発火しなかった) ので、
  // 作成側の台帳が空のまま。ここが実際の名前の出所になる。
  static const char* kStrSels[] = {"label", "functionName", "name", "debugName"};
  static const char* kObjSels[] = {"computeFunction", "function", "computeFunctionHandle"};
  @try {
    for (const char* sn : kStrSels) {
      SEL sel = sel_getUid(sn);
      if ([pso respondsToSelector:sel]) {
        NSString* v = ((NSString * (*)(id, SEL)) objc_msgSend)(pso, sel);
        if (v && [v isKindOfClass:[NSString class]] && [v length] > 0) {
          std::string t([v UTF8String]);
          std::lock_guard<std::mutex> lk(g_mu);
          g_pipe_name[key] = t;
          return t;
        }
      }
    }
    for (const char* sn : kObjSels) {
      SEL sel = sel_getUid(sn);
      if ([pso respondsToSelector:sel]) {
        id fn = ((id(*)(id, SEL))objc_msgSend)(pso, sel);
        if (fn && [fn respondsToSelector:@selector(name)]) {
          NSString* v = [fn name];
          if (v && [v length] > 0) {
            std::string t([v UTF8String]);
            std::lock_guard<std::mutex> lk(g_mu);
            g_pipe_name[key] = t;
            return t;
          }
        }
      }
    }
  } @catch (...) {
  }
  if (g_dbg_left.load(std::memory_order_relaxed) > 0) {
    g_dbg_left.fetch_sub(1, std::memory_order_relaxed);
    @try {
      Class c = object_getClass(pso);
      fprintf(stderr, "[metal_probe] 台帳に無い pipeline cls=%s desc=%s\n",
              c ? class_getName(c) : "?", [[pso description] UTF8String]);
      // 名前がどのアクセサに入っているかを探すため、引数なしのメソッドを並べる。
      for (Class k = c; k; k = class_getSuperclass(k)) {
        unsigned cnt = 0;
        Method* ms = class_copyMethodList(k, &cnt);
        std::string line;
        for (unsigned i = 0; i < cnt; ++i) {
          const char* nm = sel_getName(method_getName(ms[i]));
          if (strchr(nm, ':') == nullptr) {
            line += nm;
            line += " ";
          }
        }
        free(ms);
        fprintf(stderr, "[metal_probe]   %s: %s\n", class_getName(k), line.c_str());
        if (strcmp(class_getName(k), "NSObject") == 0) {
          break;
        }
      }
    } @catch (...) {
    }
  }
  // 台帳にも label にも無いとき: description を名前にする (ドライバの
  // pipeline state は description に関数名を含むことがある)。ポインタは毎回
  // 変わるので落とす。名前が取れないビルドでは全部が 1 つの名前に潰れるだけで、
  // 「(pre-install pipeline)」1 本と情報量は変わらない (悪化はしない)。
  @try {
    NSString* d = [pso description];
    if (d && [d length] > 0) {
      std::string t([d UTF8String]);
      for (size_t i = 0; i + 1 < t.size();) {
        if (t[i] == '0' && t[i + 1] == 'x') {
          size_t j = i + 2;
          while (j < t.size() && isxdigit(static_cast<unsigned char>(t[j]))) {
            ++j;
          }
          t.erase(i, j - i);
        } else {
          ++i;
        }
      }
      std::lock_guard<std::mutex> lk(g_mu);
      g_pipe_name[key] = t;  // 以降は台帳から引く (description の再取得を避ける)
      return t;
    }
  } @catch (...) {
  }
  return std::string("(pre-install pipeline)");
}

void note_pipe(id pso) {
  std::string nm = name_for_pipe(pso);
  std::lock_guard<std::mutex> lk(g_mu);
  t_pipe = intern_locked(nm);
}

void record_cb(std::vector<int>* names, double t0, double t1) {
  double ms = (t1 - t0) * 1000.0;
  if (!(ms > 0.0) || !(t1 > 0.0)) {
    ms = 0.0;  // GPU 時間が取れなかった CB (空の CB など)
  }
  std::lock_guard<std::mutex> lk(g_mu);
  g_cb += 1;
  g_gpu_sum_ms += ms;
  if (ms > 0.0) {
    g_ivals.emplace_back(t0, t1);
  }
  if (names && !names->empty()) {
    g_cb_with_dispatch += 1;
    double share = ms / static_cast<double>(names->size());
    for (int id : *names) {
      if (id >= 0 && id < static_cast<int>(g_gpu_ms.size())) {
        g_gpu_ms[static_cast<size_t>(id)] += share;
      }
    }
  }
}

void note_commit(id self) {
  std::vector<int>* names = t_pending_names;
  t_pending_names = nullptr;
  g_pending.fetch_add(1, std::memory_order_relaxed);
  @try {
    id<MTLCommandBuffer> cb = (id<MTLCommandBuffer>)self;
    [cb addCompletedHandler:^(id<MTLCommandBuffer> b) {
      record_cb(names, [b GPUStartTime], [b GPUEndTime]);
      delete names;
      g_pending.fetch_sub(1, std::memory_order_relaxed);
    }];
  } @catch (...) {
    delete names;
    g_pending.fetch_sub(1, std::memory_order_relaxed);
  }
}

void remember_pipe(id pso, NSString* nm) {
  if (g_dbg_left.load(std::memory_order_relaxed) > 0) {
    g_dbg_left.fetch_sub(1, std::memory_order_relaxed);
    @try {
      NSString* plbl = [pso respondsToSelector:@selector(label)] ? [pso label] : nil;
      fprintf(stderr, "[metal_probe] newPipe cls=%s name=%s pso.label=%s pso.desc=%s\n",
              object_getClass(pso) ? class_getName(object_getClass(pso)) : "?",
              nm ? [nm UTF8String] : "(nil)", plbl ? [plbl UTF8String] : "(nil)",
              [[pso description] UTF8String]);
    } @catch (...) {
    }
  }
  if (!pso || !nm || [nm length] == 0) {
    return;
  }
  std::lock_guard<std::mutex> lk(g_mu);
  g_pipe_name[static_cast<const void*>(pso)] = std::string([nm UTF8String]);
}

NSString* name_of_desc(id desc) {
  @try {
    if ([desc respondsToSelector:@selector(label)]) {
      NSString* lbl = [desc label];
      if (lbl && [lbl length] > 0) {
        return lbl;
      }
    }
    if ([desc respondsToSelector:@selector(computeFunction)]) {
      id fn = [desc computeFunction];
      if (fn) {
        return [fn name];
      }
    }
  } @catch (...) {
  }
  return nil;
}

NSString* name_of_fn(id fn) {
  @try {
    if (fn && [fn respondsToSelector:@selector(name)]) {
      return [fn name];
    }
  } @catch (...) {
  }
  return nil;
}

// ---------------------------------------------------------------------------
// 差し替えた IMP の置き場 (セレクタごとに MAXS 個まで)
// ---------------------------------------------------------------------------
struct SlotSet {
  IMP orig[MAXS] = {};
  Class cls[MAXS] = {};
  int n = 0;
};
SlotSet g_dtg, g_dth, g_dti, g_pipe_sel, g_commit, g_np;

// trampoline を MAXS 本ずつ作る。I 番目は「I 番目に差し替えたクラス」の
// 元の IMP だけを呼ぶ (上のコメントの無限再帰対策)。
template <int I>
void t_dtg(id self, SEL cmd, MTLSize a, MTLSize b) {
  Depth d(&t_d_dtg);
  if (d.outer && g_on.load(std::memory_order_relaxed)) {
    note_dispatch();
  }
  reinterpret_cast<void (*)(id, SEL, MTLSize, MTLSize)>(g_dtg.orig[I])(self, cmd, a, b);
}

template <int I>
void t_dth(id self, SEL cmd, MTLSize a, MTLSize b) {
  Depth d(&t_d_dth);
  if (d.outer && g_on.load(std::memory_order_relaxed)) {
    note_dispatch();
  }
  reinterpret_cast<void (*)(id, SEL, MTLSize, MTLSize)>(g_dth.orig[I])(self, cmd, a, b);
}

// 間接 dispatch (grid をバッファから読む形)。MLX も使う経路なので数える。
template <int I>
void t_dti(id self, SEL cmd, id buf, NSUInteger off, MTLSize tg) {
  Depth d(&t_d_dti);
  if (d.outer && g_on.load(std::memory_order_relaxed)) {
    note_dispatch();
  }
  reinterpret_cast<void (*)(id, SEL, id, NSUInteger, MTLSize)>(g_dti.orig[I])(self, cmd, buf, off,
                                                                             tg);
}

template <int I>
void t_pipe_sel(id self, SEL cmd, id pso) {
  Depth d(&t_d_pipe);
  if (d.outer && g_on.load(std::memory_order_relaxed)) {
    note_pipe(pso);
  }
  reinterpret_cast<void (*)(id, SEL, id)>(g_pipe_sel.orig[I])(self, cmd, pso);
}

template <int I>
void t_commit(id self, SEL cmd) {
  Depth d(&t_d_commit);
  if (d.outer && g_on.load(std::memory_order_relaxed)) {
    note_commit(self);
  }
  reinterpret_cast<void (*)(id, SEL)>(g_commit.orig[I])(self, cmd);
}

// `newComputePipelineStateWith...:error:` の全変種を 1 つの trampoline で受ける。
// 変種ごとに引数の数が違う (descriptor 版 4 引数、function 版 2 引数、
// compilerTaskOptions 版 3 引数 …) が、**必要なのは第 1 引数 (descriptor か
// function) と戻り値だけ**。arm64 では 6 引数まで全部レジスタ渡しなので、
// 常に 6 引数の形で受けて 6 引数の形で元へ渡せば、実際の引数が少ない変種でも
// 余分なレジスタが無視されるだけで安全に通る (スタック渡しに落ちない範囲)。
// completionHandler 版 (戻り値 void の非同期) はこの形に合わないので外す。
template <int I>
id t_np(id self, SEL cmd, id a0, void* a1, void* a2, void* a3) {
  id pso = reinterpret_cast<id (*)(id, SEL, id, void*, void*, void*)>(g_np.orig[I])(self, cmd, a0,
                                                                                    a1, a2, a3);
  if (pso) {
    NSString* nm = name_of_desc(a0);
    if (!nm) {
      nm = name_of_fn(a0);
    }
    remember_pipe(pso, nm);
  }
  return pso;
}

// 関数テンプレートはテンプレートテンプレート引数に渡せないので、MAXS 本を
// マクロで並べる (MAXS を変えるならここも直す)。
static_assert(MAXS == 48, "MP_TAB は MAXS==48 前提");
// clang-format off
#define MP_TAB(fn) {reinterpret_cast<IMP>(&fn<0>), reinterpret_cast<IMP>(&fn<1>), reinterpret_cast<IMP>(&fn<2>), reinterpret_cast<IMP>(&fn<3>), reinterpret_cast<IMP>(&fn<4>), reinterpret_cast<IMP>(&fn<5>), reinterpret_cast<IMP>(&fn<6>), reinterpret_cast<IMP>(&fn<7>), reinterpret_cast<IMP>(&fn<8>), reinterpret_cast<IMP>(&fn<9>), reinterpret_cast<IMP>(&fn<10>), reinterpret_cast<IMP>(&fn<11>), reinterpret_cast<IMP>(&fn<12>), reinterpret_cast<IMP>(&fn<13>), reinterpret_cast<IMP>(&fn<14>), reinterpret_cast<IMP>(&fn<15>), reinterpret_cast<IMP>(&fn<16>), reinterpret_cast<IMP>(&fn<17>), reinterpret_cast<IMP>(&fn<18>), reinterpret_cast<IMP>(&fn<19>), reinterpret_cast<IMP>(&fn<20>), reinterpret_cast<IMP>(&fn<21>), reinterpret_cast<IMP>(&fn<22>), reinterpret_cast<IMP>(&fn<23>), reinterpret_cast<IMP>(&fn<24>), reinterpret_cast<IMP>(&fn<25>), reinterpret_cast<IMP>(&fn<26>), reinterpret_cast<IMP>(&fn<27>), reinterpret_cast<IMP>(&fn<28>), reinterpret_cast<IMP>(&fn<29>), reinterpret_cast<IMP>(&fn<30>), reinterpret_cast<IMP>(&fn<31>), reinterpret_cast<IMP>(&fn<32>), reinterpret_cast<IMP>(&fn<33>), reinterpret_cast<IMP>(&fn<34>), reinterpret_cast<IMP>(&fn<35>), reinterpret_cast<IMP>(&fn<36>), reinterpret_cast<IMP>(&fn<37>), reinterpret_cast<IMP>(&fn<38>), reinterpret_cast<IMP>(&fn<39>), reinterpret_cast<IMP>(&fn<40>), reinterpret_cast<IMP>(&fn<41>), reinterpret_cast<IMP>(&fn<42>), reinterpret_cast<IMP>(&fn<43>), reinterpret_cast<IMP>(&fn<44>), reinterpret_cast<IMP>(&fn<45>), reinterpret_cast<IMP>(&fn<46>), reinterpret_cast<IMP>(&fn<47>)}
// clang-format on

const std::array<IMP, MAXS> TAB_DTG = MP_TAB(t_dtg);
const std::array<IMP, MAXS> TAB_DTH = MP_TAB(t_dth);
const std::array<IMP, MAXS> TAB_DTI = MP_TAB(t_dti);
const std::array<IMP, MAXS> TAB_PIPE = MP_TAB(t_pipe_sel);
const std::array<IMP, MAXS> TAB_COMMIT = MP_TAB(t_commit);
const std::array<IMP, MAXS> TAB_NP = MP_TAB(t_np);

// ---------------------------------------------------------------------------
// クラス探索と差し替え
// ---------------------------------------------------------------------------
bool has_all(Class cls, const SEL* sels, int n) {
  for (int i = 0; i < n; ++i) {
    if (!class_getInstanceMethod(cls, sels[i])) {
      return false;
    }
  }
  return true;
}

int swizzle_kind(SEL sel, const SEL* guard, int n_guard, const std::array<IMP, MAXS>& tab,
                 SlotSet& slots, Class* list, int n) {
  for (int i = 0; i < n && slots.n < MAXS; ++i) {
    Class cls = list[i];
    Method m = class_getInstanceMethod(cls, sel);
    if (!m) {
      continue;
    }
    // 継承で降りてきただけのクラスは飛ばす (親を差し替えれば足りる)。
    Class sup = class_getSuperclass(cls);
    if (sup) {
      Method ms = class_getInstanceMethod(sup, sel);
      if (ms && method_getImplementation(ms) == method_getImplementation(m)) {
        continue;
      }
    }
    // `commit` のようにありふれた名前は Metal と無関係のクラスも持っている。
    // Metal のそれだと分かる別のセレクタを一緒に持っているかで絞る。
    if (!has_all(cls, guard, n_guard)) {
      continue;
    }
    int slot = slots.n;
    slots.orig[slot] = method_getImplementation(m);
    slots.cls[slot] = cls;
    slots.n += 1;
    method_setImplementation(m, tab[static_cast<size_t>(slot)]);
  }
  return slots.n;
}

/// device のクラスが自分で実装している `newComputePipelineState...` を
/// **前置き一致で全部**差し替える。2026-09-03 の実測で、決め打ちした 3 つの
/// セレクタ (`...WithDescriptor:options:reflection:error:` など) は 1 回も
/// 発火しなかった (MLX は別の変種を使っている)。どれを使うか分からないので
/// 名前で拾う。`completionHandler` 版 (非同期・戻り値 void) は形が違うので外す。
int swizzle_new_pipeline(const SEL* guard, int n_guard, Class* list, int n) {
  for (int i = 0; i < n && g_np.n < MAXS; ++i) {
    Class cls = list[i];
    if (!has_all(cls, guard, n_guard)) {
      continue;
    }
    unsigned cnt = 0;
    Method* ms = class_copyMethodList(cls, &cnt);  // 自分で実装したものだけ
    if (!ms) {
      continue;
    }
    for (unsigned k = 0; k < cnt && g_np.n < MAXS; ++k) {
      const char* nm = sel_getName(method_getName(ms[k]));
      if (strncmp(nm, "newComputePipelineState", 23) != 0) {
        continue;
      }
      if (strstr(nm, "completionHandler")) {
        continue;
      }
      int slot = g_np.n;
      g_np.orig[slot] = method_getImplementation(ms[k]);
      g_np.cls[slot] = cls;
      g_np.n += 1;
      method_setImplementation(ms[k], TAB_NP[static_cast<size_t>(slot)]);
    }
    free(ms);
  }
  return g_np.n;
}

bool g_installed = false;
int g_hits[8] = {0, 0, 0, 0, 0, 0, 0, 0};

}  // namespace

// ---------------------------------------------------------------------------
// C ABI
// ---------------------------------------------------------------------------

/// 差し替えを入れる。戻り値は dispatch/commit 側で差し替えたクラス数 (0 なら失敗)。
/// hits[0..6] に種類ごとの内訳を書く (NULL 可)。
MP_EXPORT int mp_install(int* hits) {
  if (!g_installed) {
    int n = objc_getClassList(nullptr, 0);
    if (n <= 0) {
      return 0;
    }
    Class* list = static_cast<Class*>(malloc(sizeof(Class) * static_cast<size_t>(n)));
    if (!list) {
      return 0;
    }
    n = objc_getClassList(list, n);

    const SEL enc_guard[] = {@selector(setComputePipelineState:), @selector(endEncoding)};
    const SEL cb_guard[] = {@selector(addCompletedHandler:), @selector(GPUStartTime),
                            @selector(GPUEndTime)};
    const SEL dev_guard[] = {@selector(newCommandQueue), @selector(newBufferWithLength:options:)};

    g_hits[0] = swizzle_kind(@selector(dispatchThreadgroups:threadsPerThreadgroup:), enc_guard, 2,
                             TAB_DTG, g_dtg, list, n);
    g_hits[1] = swizzle_kind(@selector(dispatchThreads:threadsPerThreadgroup:), enc_guard, 2,
                             TAB_DTH, g_dth, list, n);
    g_hits[2] = swizzle_kind(@selector(setComputePipelineState:), enc_guard, 2, TAB_PIPE,
                             g_pipe_sel, list, n);
    g_hits[7] = swizzle_kind(
        @selector(dispatchThreadgroupsWithIndirectBuffer:indirectBufferOffset:threadsPerThreadgroup:),
        enc_guard, 2, TAB_DTI, g_dti, list, n);
    g_hits[3] = swizzle_kind(@selector(commit), cb_guard, 3, TAB_COMMIT, g_commit, list, n);
    g_hits[4] = swizzle_new_pipeline(dev_guard, 2, list, n);
    g_hits[5] = 0;
    g_hits[6] = 0;

    free(list);
    g_installed = true;
  }
  if (hits) {
    memcpy(hits, g_hits, sizeof(g_hits));
  }
  return g_hits[0] + g_hits[1] + g_hits[3];  // dispatch 2 種 + commit
}

MP_EXPORT void mp_debug(int n) { g_dbg_left.store(n, std::memory_order_relaxed); }

MP_EXPORT void mp_enable(int on) { g_on.store(on != 0, std::memory_order_relaxed); }

MP_EXPORT void mp_reset(void) {
  std::lock_guard<std::mutex> lk(g_mu);
  for (auto& c : g_counts) {
    c = 0;
  }
  for (auto& v : g_gpu_ms) {
    v = 0.0;
  }
  g_dispatch = 0;
  g_cb = 0;
  g_cb_with_dispatch = 0;
  g_gpu_sum_ms = 0.0;
  g_ivals.clear();
}

/// completion handler の消化を待つ。全部片付いたら 0、時間切れなら残数。
MP_EXPORT int mp_quiesce(int timeout_ms) {
  for (int i = 0; i < timeout_ms; ++i) {
    if (g_pending.load(std::memory_order_relaxed) <= 0) {
      return 0;
    }
    usleep(1000);
  }
  return static_cast<int>(g_pending.load(std::memory_order_relaxed));
}

/// 集計値を取り出す。gpu_union_ms は CB の GPU 区間の和集合 (重なりを 1 回だけ
/// 数えたもの)。n_names はカーネル名の種類数。
MP_EXPORT void mp_stats(uint64_t* dispatches, uint64_t* command_buffers,
                        uint64_t* cb_with_dispatch, double* gpu_sum_ms, double* gpu_union_ms,
                        int* n_names) {
  std::lock_guard<std::mutex> lk(g_mu);
  if (dispatches) {
    *dispatches = g_dispatch;
  }
  if (command_buffers) {
    *command_buffers = g_cb;
  }
  if (cb_with_dispatch) {
    *cb_with_dispatch = g_cb_with_dispatch;
  }
  if (gpu_sum_ms) {
    *gpu_sum_ms = g_gpu_sum_ms;
  }
  if (n_names) {
    *n_names = static_cast<int>(g_names.size());
  }
  if (gpu_union_ms) {
    std::vector<std::pair<double, double>> iv = g_ivals;
    std::sort(iv.begin(), iv.end());
    double total = 0.0, cur0 = 0.0, cur1 = -1.0;
    for (auto& p : iv) {
      if (cur1 < 0.0) {
        cur0 = p.first;
        cur1 = p.second;
      } else if (p.first > cur1) {
        total += cur1 - cur0;
        cur0 = p.first;
        cur1 = p.second;
      } else if (p.second > cur1) {
        cur1 = p.second;
      }
    }
    if (cur1 >= 0.0) {
      total += cur1 - cur0;
    }
    *gpu_union_ms = total * 1000.0;
  }
}

/// idx 番のカーネル名とその回数・GPU 時間 (ms)。戻り値は名前の長さ (-1 = 範囲外)。
MP_EXPORT int mp_name(int idx, char* buf, int buflen, uint64_t* count, double* gpu_ms) {
  std::lock_guard<std::mutex> lk(g_mu);
  if (idx < 0 || idx >= static_cast<int>(g_names.size())) {
    return -1;
  }
  const std::string& s = g_names[static_cast<size_t>(idx)];
  if (buf && buflen > 0) {
    int n = static_cast<int>(s.size());
    if (n > buflen - 1) {
      n = buflen - 1;
    }
    memcpy(buf, s.data(), static_cast<size_t>(n));
    buf[n] = '\0';
  }
  if (count) {
    *count = g_counts[static_cast<size_t>(idx)];
  }
  if (gpu_ms) {
    *gpu_ms = g_gpu_ms[static_cast<size_t>(idx)];
  }
  return static_cast<int>(s.size());
}

/// GPU 区間の記録数 (union の材料が何本あるか)。整合確認用。
MP_EXPORT uint64_t mp_interval_count(void) {
  std::lock_guard<std::mutex> lk(g_mu);
  return static_cast<uint64_t>(g_ivals.size());
}

/// 差し替えたクラスの名前を取り出す (kind 0..6、idx はそのセレクタの何番目か)。
/// 戻り値は名前の長さ (-1 = 範囲外)。
MP_EXPORT int mp_swizzled_class(int kind, int idx, char* buf, int buflen) {
  SlotSet* s = nullptr;
  switch (kind) {
    case 0: s = &g_dtg; break;
    case 1: s = &g_dth; break;
    case 2: s = &g_pipe_sel; break;
    case 3: s = &g_commit; break;
    case 4: s = &g_np; break;
    case 5: return -1;
    case 6: return -1;
    case 7: s = &g_dti; break;
    default: return -1;
  }
  if (idx < 0 || idx >= s->n || !s->cls[idx]) {
    return -1;
  }
  const char* nm = class_getName(s->cls[idx]);
  int n = static_cast<int>(strlen(nm));
  if (buf && buflen > 0) {
    int c = n > buflen - 1 ? buflen - 1 : n;
    memcpy(buf, nm, static_cast<size_t>(c));
    buf[c] = '\0';
  }
  return n;
}
