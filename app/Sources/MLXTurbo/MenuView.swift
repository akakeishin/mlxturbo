import SwiftUI

struct MenuView: View {
    @EnvironmentObject private var settings: Settings
    @EnvironmentObject private var server: ServerController
    @EnvironmentObject private var hub: HubClient
    @Environment(\.openWindow) private var openWindow

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            header

            Divider()

            if !hub.local.isEmpty {
                Text("モデル").font(.caption).foregroundStyle(.secondary)
                Picker("", selection: modelBinding) {
                    Text("選択なし").tag("")
                    ForEach(hub.local) { m in
                        Text(m.name).tag(m.path)
                    }
                }
                .labelsHidden()
                .pickerStyle(.menu)
            }

            HStack {
                Button(server.state == .stopped ? "起動" : "停止") {
                    server.state == .stopped ? server.start() : server.stop()
                }
                .disabled(settings.modelPath.isEmpty && server.state == .stopped)

                Button("モデルを探す…") { openWindow(id: "models") }
            }

            if case .failed(let why) = server.state {
                Text(why)
                    .font(.caption)
                    .foregroundStyle(.red)
                    .fixedSize(horizontal: false, vertical: true)
            }

            Divider()
            Button("終了") {
                server.stop()
                NSApplication.shared.terminate(nil)
            }
        }
        .padding(12)
        .frame(width: 300)
        .task { await hub.refreshLocal() }
    }

    /// モデルを選び直したら、走っているサーバーは起こし直す
    /// (v1 は実行中の載せ替えを持たない — ServerController の注記を参照)。
    private var modelBinding: Binding<String> {
        Binding(
            get: { settings.modelPath },
            set: { newValue in
                guard newValue != settings.modelPath else { return }
                settings.modelPath = newValue
                if server.state != .stopped { server.restart() }
            }
        )
    }

    @ViewBuilder private var header: some View {
        HStack {
            Circle()
                .fill(stateColor)
                .frame(width: 8, height: 8)
            Text(stateText).font(.system(size: 13, weight: .medium))
            Spacer()
        }
        if let s = server.status {
            let mem = s.rssBytes.map { String(format: "%.1fGB", Double($0) / 1_073_741_824) } ?? "—"
            Text("\(s.modelName ?? "—") · \(s.runnerKind ?? "—") · \(mem)")
                .font(.caption)
                .foregroundStyle(.secondary)
                .lineLimit(1)
        }
    }

    private var stateColor: Color {
        switch server.state {
        case .stopped: return .secondary
        case .starting: return .orange
        case .ready: return .green
        case .failed: return .red
        }
    }

    private var stateText: String {
        switch server.state {
        case .stopped: return "停止中"
        case .starting: return "読み込み中…"
        case .ready: return "待ち受け中 :\(settings.port)"
        case .failed: return "失敗"
        }
    }
}
