# Run tests

using TestItemRunner

@run_package_tests filter=t -> !(:skip in t.tags)