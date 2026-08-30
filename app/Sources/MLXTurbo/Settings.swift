import Foundation

/// ユーザーが選んだ実行パスとサーバー引数。UserDefaults に持つ。
///
/// mlxturbo 本体は Python なので、どの Python 環境の mlxturbo を使うかは
/// 機械ごとに違う。既定では PATH と、よくある置き場所を順に探す。
@MainActor
final class Settings: ObservableObject {
    @Published var executable: String {
        didSet { UserDefaults.standard.set(executable, forKey: "executable") }
    }
    @Published var modelPath: String {
        didSet { UserDefaults.standard.set(modelPath, forKey: "modelPath") }
    }
    @Published var ngramPath: String {
        didSet { UserDefaults.standard.set(ngramPath, forKey: "ngramPath") }
    }
    @Published var mtpPath: String {
        didSet { UserDefaults.standard.set(mtpPath, forKey: "mtpPath") }
    }
    @Published var port: Int {
        didSet { UserDefaults.standard.set(port, forKey: "port") }
    }

    init() {
        let d = UserDefaults.standard
        executable = d.string(forKey: "executable") ?? Settings.findExecutable() ?? ""
        modelPath = d.string(forKey: "modelPath") ?? ""
        ngramPath = d.string(forKey: "ngramPath") ?? ""
        mtpPath = d.string(forKey: "mtpPath") ?? ""
        port = d.object(forKey: "port") as? Int ?? 11235
    }

    var baseURL: URL { URL(string: "http://127.0.0.1:\(port)")! }

    /// `mlxturbo` の実行ファイルを探す。見つからなければ nil (設定で入れてもらう)。
    ///
    /// GUI アプリは Terminal と違ってシェルの初期化を通らないので PATH が細い。
    /// よくある置き場所を直接見る。
    static func findExecutable() -> String? {
        let home = FileManager.default.homeDirectoryForCurrentUser.path
        let candidates = [
            "\(home)/dev/fastmlx/.venv/bin/mlxturbo",
            "\(home)/.local/bin/mlxturbo",
            "/opt/homebrew/bin/mlxturbo",
            "/usr/local/bin/mlxturbo",
        ]
        return candidates.first { FileManager.default.isExecutableFile(atPath: $0) }
    }

    /// `mlxturbo` と同じ bin にある `mlxturbo-serve`。
    var serveExecutable: String {
        executable.isEmpty ? "" : executable + "-serve"
    }
}
