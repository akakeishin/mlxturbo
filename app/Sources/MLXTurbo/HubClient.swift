import Foundation

struct LocalModel: Decodable, Identifiable, Hashable {
    let path: String
    let name: String
    let modelType: String?
    let sizeBytes: Int?
    let hasMTP: Bool?
    let contextLength: Int?

    var id: String { path }

    enum CodingKeys: String, CodingKey {
        case path, name
        case modelType = "model_type"
        case sizeBytes = "size_bytes"
        case hasMTP = "has_mtp"
        case contextLength = "context_length"
    }
}

struct RemoteModel: Decodable, Identifiable, Hashable {
    let id: String
    let downloads: Int?
    let likes: Int?
    let alreadyLocal: Bool?

    enum CodingKeys: String, CodingKey {
        case id, downloads, likes
        case alreadyLocal = "already_local"
    }
}

/// `mlxturbo hub ...` を叩いて JSON を受け取る。
///
/// HF の検索とダウンロードを Swift で書き直すと、レジューム・分割・認証を
/// 作り直すことになるので、huggingface_hub をそのまま使える Python 側に
/// 口を置いてある。
@MainActor
final class HubClient: ObservableObject {
    @Published private(set) var local: [LocalModel] = []
    @Published private(set) var results: [RemoteModel] = []
    @Published private(set) var busy = false
    @Published private(set) var lastError: String?

    private let settings: Settings

    init(settings: Settings) { self.settings = settings }

    func refreshLocal() async {
        guard let data = await run(["hub", "list"]) else { return }
        do { local = try JSONDecoder().decode([LocalModel].self, from: data) }
        catch { lastError = "一覧を読めませんでした: \(error.localizedDescription)" }
    }

    func search(_ query: String) async {
        guard !query.trimmingCharacters(in: .whitespaces).isEmpty else {
            results = []
            return
        }
        guard let data = await run(["hub", "search", query]) else { return }
        do { results = try JSONDecoder().decode([RemoteModel].self, from: data) }
        catch { lastError = "検索結果を読めませんでした: \(error.localizedDescription)" }
    }

    private func run(_ args: [String]) async -> Data? {
        let exe = settings.executable
        guard !exe.isEmpty, FileManager.default.isExecutableFile(atPath: exe) else {
            lastError = "mlxturbo が見つかりません。設定で場所を指定してください"
            return nil
        }
        busy = true
        defer { busy = false }
        lastError = nil
        return await Task.detached {
            let p = Process()
            p.executableURL = URL(fileURLWithPath: exe)
            p.arguments = args
            let out = Pipe()
            p.standardOutput = out
            p.standardError = Pipe()
            do { try p.run() } catch { return nil }
            let data = out.fileHandleForReading.readDataToEndOfFile()
            p.waitUntilExit()
            return p.terminationStatus == 0 ? data : nil
        }.value
    }
}
