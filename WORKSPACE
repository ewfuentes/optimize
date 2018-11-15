load("@bazel_tools//tools/build_defs/repo:git.bzl", "git_repository", "new_git_repository")
load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

git_repository(
    name = "gtest",
    remote = "https://github.com/google/googletest.git",
    tag = "release-1.8.1",
)

new_git_repository(
    name = "eigen",
    remote = "https://github.com/eigenteam/eigen-git-mirror.git",
    tag = "3.3.5",
    build_file=":eigen.BUILD"
)

git_repository(
    name = "glog",
    remote = "https://github.com/google/glog.git",
    commit = "5c292672df04ab82a97be5116d47dc0cc544b39f",
    init_submodules = 1,
)

http_archive(
    name = "com_github_gflags_gflags",
    sha256 = "6e16c8bc91b1310a44f3965e616383dbda48f83e8c1eaa2370a215057b00cabe",
    strip_prefix = "gflags-77592648e3f3be87d6c7123eb81cbad75f9aef5a",
    urls = [
        "https://mirror.bazel.build/github.com/gflags/gflags/archive/77592648e3f3be87d6c7123eb81cbad75f9aef5a.tar.gz",
        "https://github.com/gflags/gflags/archive/77592648e3f3be87d6c7123eb81cbad75f9aef5a.tar.gz",
    ],
)