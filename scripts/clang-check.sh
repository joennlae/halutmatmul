#!/bin/bash
# inspiration https://github.com/jidicula/clang-format-action/blob/71fcf6478ab3cdba469c7d8092e07162f29767ec/entrypoint.sh#L1

format_diff() {
	local filepath="$1"
	local_format="$(clang-format -n --Werror --style=file --fallback-style="LLVM" "${filepath}")"
	local format_status="$?"
	if [[ "${format_status}" -ne 0 ]]; then
		echo "Failed on file: $filepath"
		echo "$local_format" >&2
		exit_code=1
		return "${format_status}"
	fi
	return 0
}

CHECK_PATH="$1"
EXCLUDE_REGEX="$2"

# Set the regex to an empty string regex if nothing was provided
if [ -z "$EXCLUDE_REGEX" ]; then
	EXCLUDE_REGEX="(lib|CMakeFiles)"
fi

echo "Exclude regex: $EXCLUDE_REGEX \nCheck path: $CHECK_PATH"

if [[ ! -d "$CHECK_PATH" ]]; then
	echo "Not a directory in the workspace, fallback to all files."
	CHECK_PATH="."
fi


# initialize exit code
exit_code=0

# All files improperly formatted will be printed to the output.
# find all C/C++ files:
#   h, H, hpp, hh, h++, hxx
#   c, C, cpp, cc, c++, cxx
c_files=$(find "$CHECK_PATH" | grep -E '\.((c|C)c?(pp|xx|\+\+)*$|(h|H)h?(pp|xx|\+\+)*$)')

# check formatting in each C file
for file in $c_files; do
	# Only check formatting if the path doesn't match the regex
	if ! [[ "${file}" =~ $EXCLUDE_REGEX ]]; then
		format_diff "${file}"
    # echo "${file}"
	fi
done

exit "$exit_code"