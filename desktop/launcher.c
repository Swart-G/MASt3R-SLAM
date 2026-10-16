#define _GNU_SOURCE

#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <poll.h>
#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/file.h>
#include <sys/stat.h>
#include <sys/statvfs.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <sys/ioctl.h>
#include <time.h>
#include <unistd.h>

#ifndef BUNDLE_VERSION
#define BUNDLE_VERSION "development"
#endif

#define PAYLOAD_MAGIC "M3SLAM1!"
#define APP_DIRECTORY "mast3r-slam-app"
#define MIN_FREE_BYTES (12ULL * 1024ULL * 1024ULL * 1024ULL)
#define LOG_LINES 256
#define LOG_WIDTH 2048

struct __attribute__((packed)) payload_footer {
    char magic[8];
    uint64_t uv_size;
    uint64_t archive_size;
};

struct ui_state {
    int enabled;
    int stage;
    int total;
    int rows;
    int cols;
    char stage_name[160];
    char app_dir[PATH_MAX];
    char lines[LOG_LINES][LOG_WIDTH];
    int line_count;
    char partial[LOG_WIDTH];
    size_t partial_len;
    FILE *log_file;
};

static struct ui_state ui;
static volatile sig_atomic_t interrupted_signal;
static volatile sig_atomic_t active_child = -1;

static void handle_signal(int signal_number) {
    interrupted_signal = signal_number;
    if (active_child > 0) kill((pid_t)active_child, signal_number);
}

static int path_join(char *output, size_t size, const char *left, const char *right) {
    int written = snprintf(output, size, "%s/%s", left, right);
    return written >= 0 && (size_t)written < size ? 0 : -1;
}

static int parent_directory(char *path) {
    char *slash = strrchr(path, '/');
    if (slash == NULL) return -1;
    if (slash == path) slash[1] = '\0';
    else *slash = '\0';
    return 0;
}

static int mkdir_p(const char *path, mode_t mode) {
    char copy[PATH_MAX];
    size_t length = strlen(path);
    if (length == 0 || length >= sizeof(copy)) return -1;
    memcpy(copy, path, length + 1);
    for (char *cursor = copy + 1; *cursor != '\0'; ++cursor) {
        if (*cursor != '/') continue;
        *cursor = '\0';
        if (mkdir(copy, mode) != 0 && errno != EEXIST) return -1;
        *cursor = '/';
    }
    if (mkdir(copy, mode) != 0 && errno != EEXIST) return -1;
    return 0;
}

static int configure_cuda_compiler(const char *app_dir) {
    char eigen[PATH_MAX], include_path[PATH_MAX * 2];
    if (path_join(eigen, sizeof(eigen), app_dir, "thirdparty/eigen") != 0) return -1;
    const char *existing_include = getenv("CPLUS_INCLUDE_PATH");
    int include_written = existing_include != NULL && existing_include[0] != '\0'
        ? snprintf(include_path, sizeof(include_path), "%s:%s", eigen, existing_include)
        : snprintf(include_path, sizeof(include_path), "%s", eigen);
    if (include_written < 0 || (size_t)include_written >= sizeof(include_path) ||
        setenv("CPLUS_INCLUDE_PATH", include_path, 1) != 0) return -1;
    if (access("/usr/bin/gcc-15", X_OK) == 0 && access("/usr/bin/g++-15", X_OK) == 0) {
        return setenv("CC", "/usr/bin/gcc-15", 1) == 0 &&
               setenv("CXX", "/usr/bin/g++-15", 1) == 0 &&
               setenv("NVCC_CCBIN", "/usr/bin/g++-15", 1) == 0 ? 0 : -1;
    }
    const char *existing = getenv("NVCC_PREPEND_FLAGS");
    char flags[4096];
    int written = existing != NULL && existing[0] != '\0'
        ? snprintf(flags, sizeof(flags), "-allow-unsupported-compiler %s", existing)
        : snprintf(flags, sizeof(flags), "-allow-unsupported-compiler");
    if (written < 0 || (size_t)written >= sizeof(flags)) return -1;
    return setenv("NVCC_PREPEND_FLAGS", flags, 1);
}

static void ui_dimensions(void) {
    struct winsize size;
    ui.rows = 24;
    ui.cols = 80;
    if (ioctl(STDOUT_FILENO, TIOCGWINSZ, &size) == 0) {
        if (size.ws_row >= 14) ui.rows = size.ws_row;
        if (size.ws_col >= 50) ui.cols = size.ws_col;
    }
}

static void ui_render(void) {
    if (!ui.enabled) return;
    ui_dimensions();
    int inner = ui.cols - 4;
    int filled = ui.total > 0 ? (inner * ui.stage) / ui.total : 0;
    dprintf(STDOUT_FILENO, "\033[H\033[2J\033[1;44;97m %-*.*s \033[0m\n", ui.cols - 2,
            ui.cols - 2, " MASt3R-SLAM · first-run setup");
    dprintf(STDOUT_FILENO, "\n  [%d/%d] %.*s\n  \033[38;5;39m", ui.stage, ui.total,
            ui.cols - 14, ui.stage_name);
    for (int index = 0; index < inner; ++index) {
        dprintf(STDOUT_FILENO, "%s", index < filled ? "█" : "░");
    }
    dprintf(STDOUT_FILENO, "\033[0m\n  Directory: %.*s\n\n  \033[1mLive log\033[0m\n", ui.cols - 15,
            ui.app_dir);

    int available = ui.rows - 10;
    int first = ui.line_count > available ? ui.line_count - available : 0;
    for (int index = first; index < ui.line_count; ++index) {
        dprintf(STDOUT_FILENO, "  %.*s\033[K\n", ui.cols - 4, ui.lines[index]);
    }
    for (int index = ui.line_count - first; index < available; ++index) {
        dprintf(STDOUT_FILENO, "\033[K\n");
    }
    dprintf(STDOUT_FILENO, "\033[7m Installation may take a while · Ctrl+C to cancel %-*s\033[0m",
            ui.cols - 55, "");
    fsync(STDOUT_FILENO);
}

static void ui_add_line(const char *line, size_t length) {
    if (length >= LOG_WIDTH) length = LOG_WIDTH - 1;
    if (ui.line_count == LOG_LINES) {
        memmove(ui.lines[0], ui.lines[1], (LOG_LINES - 1) * LOG_WIDTH);
        ui.line_count--;
    }
    char *target = ui.lines[ui.line_count++];
    size_t output = 0;
    int escape = 0;
    for (size_t index = 0; index < length && output < LOG_WIDTH - 1; ++index) {
        unsigned char value = (unsigned char)line[index];
        if (escape) {
            if (value >= 0x40 && value <= 0x7e) escape = 0;
            continue;
        }
        if (value == 0x1b) {
            escape = 1;
            continue;
        }
        if (value == '\r' || value == '\t') value = ' ';
        if (value < 0x20 && value != ' ') continue;
        target[output++] = (char)value;
    }
    target[output] = '\0';
}

static void ui_feed(const char *data, size_t length) {
    if (ui.log_file != NULL) {
        fwrite(data, 1, length, ui.log_file);
        fflush(ui.log_file);
    }
    if (!ui.enabled) {
        fwrite(data, 1, length, stdout);
        fflush(stdout);
        return;
    }
    for (size_t index = 0; index < length; ++index) {
        char value = data[index];
        if (value == '\n' || value == '\r') {
            if (ui.partial_len > 0) {
                ui_add_line(ui.partial, ui.partial_len);
                ui.partial_len = 0;
            }
        } else if (ui.partial_len < sizeof(ui.partial) - 1) {
            ui.partial[ui.partial_len++] = value;
        }
    }
    ui_render();
}

static void ui_message(const char *message) {
    ui_feed(message, strlen(message));
    ui_feed("\n", 1);
}

static void ui_stage(int stage, const char *name) {
    ui.stage = stage;
    snprintf(ui.stage_name, sizeof(ui.stage_name), "%s", name);
    if (!ui.enabled) fprintf(stdout, "\n[%d/%d] %s\n", stage, ui.total, name);
    if (ui.log_file != NULL) {
        fprintf(ui.log_file, "\n[%d/%d] %s\n", stage, ui.total, name);
        fflush(ui.log_file);
    }
    ui_render();
}

static void ui_start(const char *app_dir, FILE *log_file) {
    memset(&ui, 0, sizeof(ui));
    ui.enabled = isatty(STDOUT_FILENO);
    ui.total = 8;
    ui.log_file = log_file;
    snprintf(ui.app_dir, sizeof(ui.app_dir), "%s", app_dir);
    if (ui.enabled) {
        dprintf(STDOUT_FILENO, "\033[?1049h\033[?25l");
        ui_render();
    }
}

static void ui_stop(void) {
    if (ui.enabled) dprintf(STDOUT_FILENO, "\033[?25h\033[?1049l");
    ui.enabled = 0;
}

static int run_command(char *const argv[]) {
    if (interrupted_signal != 0) return 128 + interrupted_signal;
    int pipefd[2];
    if (pipe(pipefd) != 0) return -1;
    pid_t child = fork();
    if (child < 0) {
        close(pipefd[0]);
        close(pipefd[1]);
        return -1;
    }
    if (child == 0) {
        close(pipefd[0]);
        dup2(pipefd[1], STDOUT_FILENO);
        dup2(pipefd[1], STDERR_FILENO);
        close(pipefd[1]);
        execvp(argv[0], argv);
        dprintf(STDERR_FILENO, "cannot execute %s: %s\n", argv[0], strerror(errno));
        _exit(127);
    }
    active_child = child;
    close(pipefd[1]);
    char buffer[8192];
    for (;;) {
        ssize_t count = read(pipefd[0], buffer, sizeof(buffer));
        if (count > 0) {
            ui_feed(buffer, (size_t)count);
            continue;
        }
        if (count < 0 && errno == EINTR) continue;
        break;
    }
    close(pipefd[0]);
    int status = 0;
    while (waitpid(child, &status, 0) < 0) {
        if (errno != EINTR) return -1;
    }
    active_child = -1;
    if (WIFEXITED(status)) return WEXITSTATUS(status);
    if (WIFSIGNALED(status)) return 128 + WTERMSIG(status);
    return -1;
}

static int copy_segment(int source, off_t offset, uint64_t size, const char *destination, mode_t mode) {
    int output = open(destination, O_WRONLY | O_CREAT | O_TRUNC, mode);
    if (output < 0) return -1;
    char buffer[1024 * 1024];
    uint64_t remaining = size;
    while (remaining > 0) {
        size_t wanted = remaining < sizeof(buffer) ? (size_t)remaining : sizeof(buffer);
        ssize_t count = pread(source, buffer, wanted, offset);
        if (count <= 0) {
            close(output);
            return -1;
        }
        size_t written = 0;
        while (written < (size_t)count) {
            ssize_t result = write(output, buffer + written, (size_t)count - written);
            if (result <= 0) {
                close(output);
                return -1;
            }
            written += (size_t)result;
        }
        offset += count;
        remaining -= (uint64_t)count;
    }
    if (fsync(output) != 0 || close(output) != 0) return -1;
    return chmod(destination, mode);
}

static int extract_payload(const char *executable, const char *bootstrap_dir,
                           char *uv_path, char *archive_path) {
    int source = open(executable, O_RDONLY);
    if (source < 0) return -1;
    off_t length = lseek(source, 0, SEEK_END);
    struct payload_footer footer;
    if (length < (off_t)sizeof(footer) ||
        pread(source, &footer, sizeof(footer), length - (off_t)sizeof(footer)) != (ssize_t)sizeof(footer) ||
        memcmp(footer.magic, PAYLOAD_MAGIC, sizeof(footer.magic)) != 0) {
        close(source);
        errno = ENOEXEC;
        return -1;
    }
    uint64_t payload_size = footer.uv_size + footer.archive_size + sizeof(footer);
    if ((uint64_t)length < payload_size) {
        close(source);
        errno = ENOEXEC;
        return -1;
    }
    off_t uv_offset = length - (off_t)payload_size;
    off_t archive_offset = uv_offset + (off_t)footer.uv_size;
    if (path_join(uv_path, PATH_MAX, bootstrap_dir, "uv") != 0 ||
        path_join(archive_path, PATH_MAX, bootstrap_dir, "application.zip") != 0 ||
        copy_segment(source, uv_offset, footer.uv_size, uv_path, 0755) != 0 ||
        copy_segment(source, archive_offset, footer.archive_size, archive_path, 0644) != 0) {
        close(source);
        return -1;
    }
    return close(source);
}

static int marker_matches(const char *app_dir) {
    char marker[PATH_MAX];
    if (path_join(marker, sizeof(marker), app_dir, ".installed") != 0) return 0;
    FILE *file = fopen(marker, "r");
    if (file == NULL) return 0;
    char version[128] = {0};
    int matched = fgets(version, sizeof(version), file) != NULL;
    fclose(file);
    version[strcspn(version, "\r\n")] = '\0';
    return matched && strcmp(version, BUNDLE_VERSION) == 0;
}

static int write_marker(const char *app_dir) {
    char marker[PATH_MAX];
    if (path_join(marker, sizeof(marker), app_dir, ".installed") != 0) return -1;
    FILE *file = fopen(marker, "w");
    if (file == NULL) return -1;
    int ok = fprintf(file, "%s\n", BUNDLE_VERSION) > 0 && fflush(file) == 0 && fsync(fileno(file)) == 0;
    return fclose(file) == 0 && ok ? 0 : -1;
}

static int install_application(const char *executable, char *app_dir) {
    char bootstrap_dir[PATH_MAX], uv_path[PATH_MAX], archive_path[PATH_MAX];
    char python[PATH_MAX], log_path[PATH_MAX], lock_path[PATH_MAX], cache[PATH_MAX], python_dir[PATH_MAX];
    if (mkdir_p(app_dir, 0755) != 0 ||
        path_join(bootstrap_dir, sizeof(bootstrap_dir), app_dir, ".bootstrap") != 0 ||
        mkdir_p(bootstrap_dir, 0755) != 0 ||
        path_join(log_path, sizeof(log_path), app_dir, "install.log") != 0 ||
        path_join(lock_path, sizeof(lock_path), app_dir, ".install.lock") != 0 ||
        path_join(python, sizeof(python), app_dir, "venv/bin/python") != 0 ||
        path_join(cache, sizeof(cache), app_dir, ".cache/uv") != 0 ||
        path_join(python_dir, sizeof(python_dir), app_dir, "python") != 0) return -1;

    int lock = open(lock_path, O_WRONLY | O_CREAT, 0644);
    if (lock < 0 || flock(lock, LOCK_EX) != 0) return -1;
    if (marker_matches(app_dir) && access(python, X_OK) == 0) {
        close(lock);
        return 0;
    }
    FILE *log_file = fopen(log_path, "a");
    if (log_file == NULL) {
        close(lock);
        return -1;
    }
    time_t now = time(NULL);
    fprintf(log_file, "\n=== setup %s · %s", BUNDLE_VERSION, ctime(&now));
    ui_start(app_dir, log_file);

    int result = 1;
    ui_stage(1, "Checking the host and unpacking bootstrap tools");
    struct statvfs filesystem;
    if (statvfs(app_dir, &filesystem) == 0) {
        uint64_t free_bytes = (uint64_t)filesystem.f_bavail * filesystem.f_frsize;
        if (free_bytes < MIN_FREE_BYTES) {
            char message[256];
            snprintf(message, sizeof(message), "Not enough free space: %.1f GiB available, 12 GiB required.",
                     free_bytes / 1024.0 / 1024.0 / 1024.0);
            ui_message(message);
            goto done;
        }
    }
    if (extract_payload(executable, bootstrap_dir, uv_path, archive_path) != 0) {
        ui_message("The executable has no valid embedded application payload.");
        goto done;
    }
    ui_message("Embedded uv and application archive are ready.");
    char *compiler_check[] = {"c++", "--version", NULL};
    char *cuda_check[] = {"nvcc", "--version", NULL};
    char *driver_check[] = {"nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader", NULL};
    if (run_command(compiler_check) != 0) {
        ui_message("A C++ compiler is required (install your distribution's build-essential package). ");
        goto done;
    }
    if (run_command(cuda_check) != 0) {
        ui_message("The NVIDIA CUDA toolkit is required; nvcc was not found.");
        goto done;
    }
    if (run_command(driver_check) != 0) {
        ui_message("A working NVIDIA driver and visible CUDA GPU are required.");
        goto done;
    }
    if (setenv("UV_PYTHON_INSTALL_DIR", python_dir, 1) != 0 ||
        setenv("UV_CACHE_DIR", cache, 1) != 0 || setenv("UV_LINK_MODE", "copy", 1) != 0 ||
        setenv("UV_NO_PROGRESS", "1", 1) != 0 || setenv("PYTHONUNBUFFERED", "1", 1) != 0 ||
        configure_cuda_compiler(app_dir) != 0) goto done;

    ui_stage(2, "Installing a private Python 3.11 runtime");
    char *python_install[] = {uv_path, "python", "install", "3.11", "--install-dir", python_dir,
                              "--no-bin", "--color", "never", NULL};
    if (run_command(python_install) != 0) goto done;

    ui_stage(3, "Creating the virtual environment");
    char venv_path[PATH_MAX];
    if (path_join(venv_path, sizeof(venv_path), app_dir, "venv") != 0) goto done;
    if (access(python, X_OK) == 0) {
        ui_message("Existing virtual environment will be reused.");
    } else {
        char *venv_create[] = {uv_path, "venv", venv_path, "--python", "3.11", "--managed-python",
                               "--seed", "--allow-existing", "--color", "never", NULL};
        if (run_command(venv_create) != 0) goto done;
    }

    ui_stage(4, "Unpacking the application sources");
    char *unpack[] = {python, "-m", "zipfile", "-e", archive_path, app_dir, NULL};
    if (run_command(unpack) != 0) goto done;

    ui_stage(5, "Installing PyTorch with the matching GPU backend");
    char *torch_install[] = {uv_path, "pip", "install", "--python", python, "--torch-backend", "auto",
                             "torch==2.9.1", "torchvision==0.24.1", "setuptools==70.0.0", "wheel", "ninja", NULL};
    if (run_command(torch_install) != 0) goto done;
    char build_preparer[PATH_MAX];
    if (path_join(build_preparer, sizeof(build_preparer), app_dir,
                  "desktop/prepare_build_environment.py") != 0) goto done;
    char *prepare_build[] = {python, build_preparer, app_dir, NULL};
    if (run_command(prepare_build) != 0) goto done;

    ui_stage(6, "Building and installing all application dependencies");
    char mast3r[PATH_MAX], in3d[PATH_MAX], project[PATH_MAX + 16];
    if (path_join(mast3r, sizeof(mast3r), app_dir, "thirdparty/mast3r") != 0 ||
        path_join(in3d, sizeof(in3d), app_dir, "thirdparty/in3d") != 0 ||
        snprintf(project, sizeof(project), "%s[viewer]", app_dir) >= (int)sizeof(project)) goto done;
    char *app_install[] = {uv_path, "pip", "install", "--python", python, "--no-build-isolation",
                           "--editable", mast3r, in3d, project, NULL};
    if (run_command(app_install) != 0) goto done;

    ui_stage(7, "Downloading and verifying model checkpoints");
    char downloader[PATH_MAX], checkpoints[PATH_MAX];
    if (path_join(downloader, sizeof(downloader), app_dir, "desktop/download_checkpoints.py") != 0 ||
        path_join(checkpoints, sizeof(checkpoints), app_dir, "checkpoints") != 0) goto done;
    char *download[] = {python, downloader, checkpoints, NULL};
    if (run_command(download) != 0) goto done;

    ui_stage(8, "Verifying CUDA and the installed application");
    char *verify[] = {python, "-c",
        "import torch, mast3r.utils.path_to_dust3r, dust3r, mast3r_slam_backends, mast3r_slam.cli; "
        "assert torch.cuda.is_available(), 'NVIDIA CUDA GPU is not available'; "
        "print('Python:', __import__('sys').version.split()[0]); "
        "print('PyTorch:', torch.__version__); print('GPU:', torch.cuda.get_device_name(0)); "
        "print('Installation verified successfully.')", NULL};
    if (run_command(verify) != 0 || write_marker(app_dir) != 0) goto done;
    result = 0;

done:
    ui_stop();
    fclose(log_file);
    flock(lock, LOCK_UN);
    close(lock);
    if (result == 0) {
        fprintf(stdout, "MASt3R-SLAM is installed in %s\n", app_dir);
    } else {
        fprintf(stderr, "MASt3R-SLAM setup failed. Full log: %s\n", log_path);
    }
    return result;
}

int main(int argc, char **argv) {
    char executable[PATH_MAX], executable_dir[PATH_MAX], app_dir[PATH_MAX], python[PATH_MAX];
    ssize_t length = readlink("/proc/self/exe", executable, sizeof(executable) - 1);
    if (length < 0 || length >= (ssize_t)sizeof(executable) - 1) {
        fprintf(stderr, "mast3r-slam: cannot resolve executable path\n");
        return 2;
    }
    executable[length] = '\0';
    struct sigaction action;
    memset(&action, 0, sizeof(action));
    action.sa_handler = handle_signal;
    sigemptyset(&action.sa_mask);
    sigaction(SIGINT, &action, NULL);
    sigaction(SIGTERM, &action, NULL);
    snprintf(executable_dir, sizeof(executable_dir), "%s", executable);
    if (parent_directory(executable_dir) != 0) return 2;
    const char *override = getenv("MAST3R_SLAM_APP_DIR");
    if (override != NULL && override[0] != '\0') {
        if (override[0] != '/') {
            fprintf(stderr, "mast3r-slam: MAST3R_SLAM_APP_DIR must be an absolute path\n");
            return 2;
        }
        if (snprintf(app_dir, sizeof(app_dir), "%s", override) >= (int)sizeof(app_dir)) return 2;
    } else if (path_join(app_dir, sizeof(app_dir), executable_dir, APP_DIRECTORY) != 0) {
        fprintf(stderr, "mast3r-slam: application path is too long\n");
        return 2;
    }
    if (argc == 2 && strcmp(argv[1], "--print-install-dir") == 0) {
        puts(app_dir);
        return 0;
    }
    if (argc == 2 && strcmp(argv[1], "--bootstrap-version") == 0) {
        puts(BUNDLE_VERSION);
        return 0;
    }
    int install_only = argc == 2 && strcmp(argv[1], "--install-only") == 0;
    if (path_join(python, sizeof(python), app_dir, "venv/bin/python") != 0) return 2;
    if (!marker_matches(app_dir) || access(python, X_OK) != 0) {
        if (install_application(executable, app_dir) != 0) return 1;
    }
    if (install_only) return 0;
    if (chdir(app_dir) != 0) {
        fprintf(stderr, "mast3r-slam: cannot enter %s: %s\n", app_dir, strerror(errno));
        return 2;
    }
    char **child_argv = calloc((size_t)argc + 3, sizeof(char *));
    if (child_argv == NULL) return 2;
    child_argv[0] = python;
    child_argv[1] = "-m";
    child_argv[2] = "mast3r_slam.cli";
    for (int index = 1; index < argc; ++index) child_argv[index + 2] = argv[index];
    child_argv[argc + 2] = NULL;
    execv(python, child_argv);
    fprintf(stderr, "mast3r-slam: failed to start application: %s\n", strerror(errno));
    free(child_argv);
    return 127;
}
