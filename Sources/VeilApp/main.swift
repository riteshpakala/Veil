//
//  main.swift
//  VeilApp
//
//  Launched directly via NSApplication (no storyboard, no @main App). Top-level code runs
//  on the main thread; `assumeIsolated` states that to the compiler. `delegate` lives for
//  the duration of `run()` — NSApplication holds its delegate weakly.
//

import AppKit

MainActor.assumeIsolated {
    let app = NSApplication.shared
    let delegate = AppDelegate()
    app.delegate = delegate
    app.setActivationPolicy(.regular)
    app.run()
}
