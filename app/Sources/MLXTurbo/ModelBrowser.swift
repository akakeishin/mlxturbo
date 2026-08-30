import SwiftUI

struct ModelBrowser: View {
    @EnvironmentObject private var settings: Settings
    @EnvironmentObject private var server: ServerController
    @EnvironmentObject private var hub: HubClient
    @State private var query = ""

    var body: some View {
        VStack(spacing: 0) {
            HStack {
                TextField("Hugging Face を検索 (例: qwen3 mlx)", text: $query)
                    .textFieldStyle(.roundedBorder)
                    .onSubmit { Task { await hub.search(query) } }
                Button("検索") { Task { await hub.search(query) } }
                    .disabled(hub.busy)
            }
            .padding(12)

            if let err = hub.lastError {
                Text(err).font(.caption).foregroundStyle(.red)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .padding(.horizontal, 12)
            }

            List {
                Section("この機械にあるもの") {
                    ForEach(hub.local) { m in
                        HStack {
                            VStack(alignment: .leading, spacing: 2) {
                                Text(m.name)
                                Text(subtitle(m)).font(.caption).foregroundStyle(.secondary)
                            }
                            Spacer()
                            if settings.modelPath == m.path {
                                Text("使用中").font(.caption).foregroundStyle(.green)
                            } else {
                                Button("使う") {
                                    settings.modelPath = m.path
                                    if server.state != .stopped { server.restart() }
                                }
                            }
                        }
                    }
                }
                if !hub.results.isEmpty {
                    Section("検索結果") {
                        ForEach(hub.results) { r in
                            HStack {
                                VStack(alignment: .leading, spacing: 2) {
                                    Text(r.id)
                                    Text("↓ \(r.downloads ?? 0)  ♡ \(r.likes ?? 0)")
                                        .font(.caption).foregroundStyle(.secondary)
                                }
                                Spacer()
                                if r.alreadyLocal == true {
                                    Text("取得済み").font(.caption).foregroundStyle(.secondary)
                                }
                            }
                        }
                    }
                }
            }
        }
        .task { await hub.refreshLocal() }
    }

    private func subtitle(_ m: LocalModel) -> String {
        var parts: [String] = []
        if let t = m.modelType { parts.append(t) }
        if let b = m.sizeBytes { parts.append(String(format: "%.0fGB", Double(b) / 1_073_741_824)) }
        if m.hasMTP == true { parts.append("MTP") }
        return parts.joined(separator: " · ")
    }
}
