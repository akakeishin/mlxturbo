import SwiftUI

@main
struct MLXTurboApp: App {
    @StateObject private var settings = Settings()
    @StateObject private var server: ServerController
    @StateObject private var hub: HubClient

    init() {
        let s = Settings()
        _settings = StateObject(wrappedValue: s)
        _server = StateObject(wrappedValue: ServerController(settings: s))
        _hub = StateObject(wrappedValue: HubClient(settings: s))
    }

    var body: some Scene {
        MenuBarExtra {
            MenuView()
                .environmentObject(settings)
                .environmentObject(server)
                .environmentObject(hub)
        } label: {
            Image(systemName: iconName)
        }
        .menuBarExtraStyle(.window)

        Window("モデルを探す", id: "models") {
            ModelBrowser()
                .environmentObject(settings)
                .environmentObject(server)
                .environmentObject(hub)
        }
        .defaultSize(width: 620, height: 460)
    }

    private var iconName: String {
        switch server.state {
        case .stopped: return "bolt.slash"
        case .starting: return "bolt.badge.clock"
        case .ready: return "bolt.fill"
        case .failed: return "exclamationmark.triangle"
        }
    }
}
