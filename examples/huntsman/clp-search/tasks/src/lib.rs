//! TDL package that runs KQL searches over CLP archives via the `clp-s` C-API library.

mod task_decl {
    use std::{
        ffi::{CStr, CString},
        fs,
        os::raw::{c_char, c_int, c_void},
        sync::Once,
        time::Instant,
    };

    use spider_tdl::{TaskContext, TdlError, task};

    // FFI to `libclp-s.so` (linked via `build.rs`).
    unsafe extern "C" {
        /// Searches the single clp-s archive at `archive_path` with the KQL `query`, invoking
        /// `callback` once per matching record. Returns 0 on success, non-zero on failure.
        fn clp_s_search_archive(
            archive_path: *const c_char,
            query: *const c_char,
            callback: Option<unsafe extern "C" fn(*const c_char, *mut c_void)>,
            user_data: *mut c_void,
        ) -> c_int;
    }

    /// C-ABI result callback: appends a matching record's JSON to the caller's result buffer.
    ///
    /// `user_data` is the `*mut Vec<u8>` passed to [`clp_s_search_archive`]. `message` is a
    /// NUL-terminated JSON record that already ends with a newline, so it is appended verbatim (no
    /// extra separator). `clp_s_search_archive` invokes this serially for a single archive, so the
    /// unsynchronized append is sound.
    unsafe extern "C" fn append_result(message: *const c_char, user_data: *mut c_void) {
        if message.is_null() || user_data.is_null() {
            return;
        }
        let buffer = unsafe { &mut *user_data.cast::<Vec<u8>>() };
        buffer.extend_from_slice(unsafe { CStr::from_ptr(message) }.to_bytes());
    }

    /// Guards one-time installation of this package's tracing subscriber.
    static LOG_INIT: Once = Once::new();

    /// Installs a package-local tracing subscriber exactly once.
    ///
    /// This TDL package is a `cdylib` with its own copy of `tracing`'s global dispatcher, distinct
    /// from the task executor that `dlopen`s it, so the executor's subscriber never observes events
    /// emitted here. This installs a subscriber owned by the package that writes JSON to stderr --
    /// the same stream the executor redirects to `em-logs/<em_id>-<executor_id>.log` -- and honors
    /// `RUST_LOG` (propagated from the execution manager). `try_init` makes a redundant call on a
    /// later task invocation a no-op.
    fn init_task_logging() {
        LOG_INIT.call_once(|| {
            let _ = tracing_subscriber::fmt()
                .event_format(tracing_subscriber::fmt::format().with_target(false).json())
                .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
                .with_ansi(false)
                .with_writer(std::io::stderr)
                .try_init();
        });
    }

    /// Runs the KQL `query` over the CLP archive at `archive_path` via the `clp-s` C-API, buffering
    /// matching records in memory and writing them as JSONL to `output_path`.
    ///
    /// The buffered results are written to `output_path` only when at least one record matched; an
    /// empty result set produces no output file.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    ///
    /// * [`TdlError::ExecutionError`] if `archive_path` or `query` contains an interior NUL byte,
    ///   the `clp_s_search_archive` call reports a failure, or the output file cannot be written.
    #[task(name = "clp_search::search")]
    pub fn search(
        ctx: TaskContext,
        archive_path: String,
        query: String,
        output_path: String,
    ) -> Result<(), TdlError> {
        init_task_logging();

        let archive_path_c = CString::new(archive_path.as_str()).map_err(|error| {
            TdlError::ExecutionError(format!("archive path `{archive_path}` contains a NUL: {error}"))
        })?;
        let query_c = CString::new(query.as_str()).map_err(|error| {
            TdlError::ExecutionError(format!("query `{query}` contains a NUL: {error}"))
        })?;

        // Benchmark instrumentation: time only the `clp-s` library search.
        let mut results: Vec<u8> = Vec::new();
        let clp_s_start = Instant::now();
        let return_code = unsafe {
            clp_s_search_archive(
                archive_path_c.as_ptr(),
                query_c.as_ptr(),
                Some(append_result),
                (&raw mut results).cast(),
            )
        };
        let clp_s_elapsed_us = u64::try_from(clp_s_start.elapsed().as_micros()).unwrap_or(u64::MAX);
        tracing::info!(
            clp_s_elapsed_us,
            job_id = ? ctx.job_id,
            task_id = ? ctx.task_id,
            "clp-s library search finished."
        );

        if return_code != 0 {
            return Err(TdlError::ExecutionError(format!(
                "`clp_s_search_archive` failed (return code {return_code}) for archive \
                 `{archive_path}`"
            )));
        }

        // No matches -> no output file, per the task contract.
        if results.is_empty() {
            return Ok(());
        }

        fs::write(&output_path, &results).map_err(|error| {
            TdlError::ExecutionError(format!(
                "failed to write output file `{output_path}` for archive `{archive_path}`: {error}"
            ))
        })?;
        Ok(())
    }
}

spider_tdl::register_tdl_package! {
    package_name: "clp_search",
    tasks: [
        task_decl::search
    ],
}
