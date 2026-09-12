//
//  AppDelegate.swift
//  VeilApp
//

import AppKit
import SwiftUI
import VeilFlux2

@MainActor
final class AppDelegate: NSObject, NSApplicationDelegate {
    private var window: NSWindow?
    let model = VeilViewModel()

    func applicationDidFinishLaunching(_ notification: Notification) {
        // Composition root: executor families available to this app.
        Flux2Registration.register()
        NSApp.mainMenu = MainMenu.build(target: self)
        let window = NSWindow(contentRect: NSRect(x: 0, y: 0, width: 1180, height: 800),
                              styleMask: [.titled, .closable, .miniaturizable, .resizable],
                              backing: .buffered, defer: false)
        window.title = "Veil"
        window.minSize = NSSize(width: 900, height: 600)
        window.contentView = NSHostingView(rootView: VeilView(model: model))
        window.center()
        window.setFrameAutosaveName("VeilMainWindow")
        window.makeKeyAndOrderFront(nil)
        self.window = window
        NSApp.activate()
    }

    func applicationShouldTerminateAfterLastWindowClosed(_ sender: NSApplication) -> Bool { true }

    @objc func addPhotos(_ sender: Any?) {
        guard let person = model.people.first else { return }
        model.choosePhotos(for: person)
    }

    @objc func exportReport(_ sender: Any?) { model.exportReport() }
    @objc func generateGuard(_ sender: Any?) { model.start(guarding: true) }
    @objc func buildGuard(_ sender: Any?) { model.startProtect() }
}
