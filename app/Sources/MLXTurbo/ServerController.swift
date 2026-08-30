import Foundation

struct ServerStatus: Decodable {
    var modelName: String?
    var runnerKind: String?
    var fallbackReason: String?
    var rssBytes: Int?
    var peakMemoryBytes: Int?
    var uptimeS: Double?
    var nSessions: Int?
    var queueDepth: Int?

    enum CodingKeys: String, CodingKey {
        case modelName = "model_name"
        case runnerKind = "runner_kind"
        case fallbackReason = "fallback_reason"
        case rssBytes = "rss_bytes"
        case peakMemoryBytes = "peak_memory_bytes"
        case uptimeS = "uptime_s"
        case nSessions = "n_sessions"
        case queueDepth = "queue_depth"
    }
}

enum ServerState: Equatable {
    case stopped
    case starting
    case ready
    case failed(String)
}

/// mlxturbo-serve のプロセスを起こし、状態を見張る。
///
/// モデルの切り替えは、走っているプロセスを止めて別の --model で起こし直す
/// 形にしている。実行中の載せ替えは in-flight のリクエストと executor
/// スレッドを跨ぐので、v1 では持ち込まない。読み込みは 12 秒程度。
@MainActor
final class ServerController: ObservableObject {
    @Published private(set) var state: ServerState = .stopped
    @Published private(set) var status: ServerStatus?
    /// 起動時のログ。失敗したときに何が起きたかを見せるため末尾だけ持つ。
    @Published private(set) var recentLog: [String] = []

    private var process: Process?
    private var pollTask: Task<Void, Never>?
    private let settings: Settings

    init(settings: Settings) { self.settings = settings }

    func start() {
        guard process == nil else { return }
        let exe = settings.serveExecutable
        guard !exe.isEmpty, FileManager.default.isExecutableFile(atPath: exe) else {
            state = .failed("mlxturbo-serve が見つかりません。設定で場所を指定してください")
            return
        }
        guard !settings.modelPath.isEmpty else {
            state = .failed("モデルが選ばれていません")
            return
        }

        var args = ["--model", settings.modelPath,
                    "--host", "127.0.0.1",
                    "--port", String(settings.port)]
        if !settings.ngramPath.isEmpty { args += ["--ngram", settings.ngramPath] }
        if !settings.mtpPath.isEmpty { args += ["--mtp", settings.mtpPath] }

        let p = Process()
        p.executableURL = URL(fileURLWithPath: exe)
        p.arguments = args
        let pipe = Pipe()
        p.standardOutput = pipe
        p.standardError = pipe
        recentLog = []
        pipe.fileHandleForReading.readabilityHandler = { [weak self] h in
            let data = h.availableData
            guard !data.isEmpty, let text = String(data: data, encoding: .utf8) else { return }
            Task { @MainActor in self?.appendLog(text) }
        }
        p.terminationHandler = { [weak self] proc in
            Task { @MainActor in self?.processDidExit(proc) }
        }

        do {
            try p.run()
        } catch {
            state = .failed("起動できませんでした: \(error.localizedDescription)")
            return
        }
        process = p
        state = .starting
        startPolling()
    }

    func stop() {
        pollTask?.cancel()
        pollTask = nil
        if let p = process, p.isRunning {
            // SIGTERM。uvicorn は これで graceful に閉じる。
            p.terminate()
        }
        process = nil
        state = .stopped
        status = nil
    }

    /// 走っていれば止めてから、選び直したモデルで起こし直す。
    func restart() {
        stop()
        start()
    }

    private func appendLog(_ text: String) {
        let lines = text.split(separator: "\n", omittingEmptySubsequences: true).map(String.init)
        recentLog.append(contentsOf: lines)
        if recentLog.count > 60 { recentLog.removeFirst(recentLog.count - 60) }
    }

    private func processDidExit(_ proc: Process) {
        guard proc === process else { return }   // stop() 済みの古いプロセス
        process = nil
        pollTask?.cancel()
        pollTask = nil
        status = nil
        if case .stopped = state { return }
        let tail = recentLog.suffix(3).joined(separator: " / ")
        state = .failed("終了しました (code \(proc.terminationStatus))" + (tail.isEmpty ? "" : ": \(tail)"))
    }

    private func startPolling() {
        pollTask?.cancel()
        pollTask = Task { [weak self] in
            while !Task.isCancelled {
                await self?.poll()
                try? await Task.sleep(for: .seconds(2))
            }
        }
    }

    private func poll() async {
        let url = settings.baseURL.appendingPathComponent("api/status")
        var req = URLRequest(url: url)
        req.timeoutInterval = 5
        do {
            let (data, resp) = try await URLSession.shared.data(for: req)
            guard (resp as? HTTPURLResponse)?.statusCode == 200 else { return }
            let s = try JSONDecoder().decode(ServerStatus.self, from: data)
            status = s
            if state == .starting { state = .ready }
        } catch {
            // 起動中はまだ口が開いていないので、失敗は無視して次の周回を待つ。
            // プロセスが死んだ場合は terminationHandler が拾う。
        }
    }
}
