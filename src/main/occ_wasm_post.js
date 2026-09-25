// Appended to the occ.wasm CLI module (--post-js), inside its factory scope.
//
// The CLI is linked with PROXY_TO_PTHREAD and EXIT_RUNTIME, so it behaves like
// an executable: main() runs on a worker thread, and when it returns the
// runtime exits and reports the status through Module.onExit. (On the JS
// thread, a busy main() never serviced the thread creations oneTBB's workers
// request, so runs of 6 or more threads fell back to running serially.)
// callMain now returns before main() has started, so Module.runMain gives
// callers the completion back. One module per command.

{
  const startMain = callMain;
  let started = false;

  /**
   * Run the occ CLI with `args` (without the program name).
   * Resolves with main()'s exit status once the runtime has exited, by which
   * time all output has reached `print`/`printErr` and its files are in FS.
   * Rejects if the module aborts. A module runs one command: create a fresh
   * one for the next.
   */
  Module["runMain"] = (args = []) =>
    new Promise((resolve, reject) => {
      if (started) {
        reject(
          new Error(
            "occ runs one command per module, like an executable: create a " +
              "new module for the next command"
          )
        );
        return;
      }
      started = true;
      const onExit = Module["onExit"];
      const onAbort = Module["onAbort"];
      // Under Node the runtime's exit also sets process.exitCode, right after
      // onExit returns. That suits a standalone executable, but not a module
      // run inside a larger program, which gets the status from the promise:
      // put it back before anything awaiting runMain resumes.
      const hostExitCode = ENVIRONMENT_IS_NODE ? process.exitCode : undefined;
      Module["onExit"] = (status) => {
        onExit?.(status);
        if (ENVIRONMENT_IS_NODE)
          queueMicrotask(() => {
            process.exitCode = hostExitCode;
          });
        resolve(status);
      };
      Module["onAbort"] = (what) => {
        onAbort?.(what);
        reject(new Error(`occ aborted: ${what}`));
      };
      try {
        startMain([...args]);
      } catch (e) {
        // With PROXY_TO_PTHREAD, callMain only starts main(); anything thrown
        // here is a failure to start it.
        reject(e);
      }
    });

  // callMain would return 0 at once, and a caller reading its result as the
  // exit status (as every caller did) would collect outputs that do not exist
  // yet. Fail loudly instead.
  Module["callMain"] = () => {
    throw new Error(
      "occ's CLI runs main() on a worker thread, so callMain cannot report " +
        "when it finishes: use `await Module.runMain(args)` instead"
    );
  };
}
