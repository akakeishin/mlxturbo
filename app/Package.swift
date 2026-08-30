// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "MLXTurbo",
    platforms: [.macOS(.v14)],
    targets: [
        .executableTarget(
            name: "MLXTurbo",
            path: "Sources/MLXTurbo",
            // .v5: MenuBarExtra + Process straddle actor boundaries in ways
            // Swift 6's strict checking rejects wholesale. The app is
            // @MainActor throughout with work handed to Task.detached
            // explicitly, so the v5 rules are what this code is written to.
            swiftSettings: [.swiftLanguageMode(.v5)]
        )
    ]
)
