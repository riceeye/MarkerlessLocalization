# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.16

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/apollolab/live-pose/FoundationPose/mycpp

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/apollolab/live-pose/FoundationPose/mycpp/build

# Include any dependencies generated for this target.
include CMakeFiles/mycpp.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/mycpp.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/mycpp.dir/flags.make

CMakeFiles/mycpp.dir/src/app/pybind_api.cpp.o: CMakeFiles/mycpp.dir/flags.make
CMakeFiles/mycpp.dir/src/app/pybind_api.cpp.o: ../src/app/pybind_api.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/apollolab/live-pose/FoundationPose/mycpp/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/mycpp.dir/src/app/pybind_api.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/mycpp.dir/src/app/pybind_api.cpp.o -c /home/apollolab/live-pose/FoundationPose/mycpp/src/app/pybind_api.cpp

CMakeFiles/mycpp.dir/src/app/pybind_api.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/mycpp.dir/src/app/pybind_api.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/apollolab/live-pose/FoundationPose/mycpp/src/app/pybind_api.cpp > CMakeFiles/mycpp.dir/src/app/pybind_api.cpp.i

CMakeFiles/mycpp.dir/src/app/pybind_api.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/mycpp.dir/src/app/pybind_api.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/apollolab/live-pose/FoundationPose/mycpp/src/app/pybind_api.cpp -o CMakeFiles/mycpp.dir/src/app/pybind_api.cpp.s

CMakeFiles/mycpp.dir/src/Utils.cpp.o: CMakeFiles/mycpp.dir/flags.make
CMakeFiles/mycpp.dir/src/Utils.cpp.o: ../src/Utils.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/apollolab/live-pose/FoundationPose/mycpp/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object CMakeFiles/mycpp.dir/src/Utils.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/mycpp.dir/src/Utils.cpp.o -c /home/apollolab/live-pose/FoundationPose/mycpp/src/Utils.cpp

CMakeFiles/mycpp.dir/src/Utils.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/mycpp.dir/src/Utils.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/apollolab/live-pose/FoundationPose/mycpp/src/Utils.cpp > CMakeFiles/mycpp.dir/src/Utils.cpp.i

CMakeFiles/mycpp.dir/src/Utils.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/mycpp.dir/src/Utils.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/apollolab/live-pose/FoundationPose/mycpp/src/Utils.cpp -o CMakeFiles/mycpp.dir/src/Utils.cpp.s

# Object files for target mycpp
mycpp_OBJECTS = \
"CMakeFiles/mycpp.dir/src/app/pybind_api.cpp.o" \
"CMakeFiles/mycpp.dir/src/Utils.cpp.o"

# External object files for target mycpp
mycpp_EXTERNAL_OBJECTS =

mycpp.cpython-38-x86_64-linux-gnu.so: CMakeFiles/mycpp.dir/src/app/pybind_api.cpp.o
mycpp.cpython-38-x86_64-linux-gnu.so: CMakeFiles/mycpp.dir/src/Utils.cpp.o
mycpp.cpython-38-x86_64-linux-gnu.so: CMakeFiles/mycpp.dir/build.make
mycpp.cpython-38-x86_64-linux-gnu.so: /usr/lib/x86_64-linux-gnu/libboost_system.so.1.71.0
mycpp.cpython-38-x86_64-linux-gnu.so: /usr/lib/x86_64-linux-gnu/libboost_program_options.so.1.71.0
mycpp.cpython-38-x86_64-linux-gnu.so: CMakeFiles/mycpp.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/apollolab/live-pose/FoundationPose/mycpp/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Linking CXX shared module mycpp.cpython-38-x86_64-linux-gnu.so"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/mycpp.dir/link.txt --verbose=$(VERBOSE)
	/usr/bin/strip /home/apollolab/live-pose/FoundationPose/mycpp/build/mycpp.cpython-38-x86_64-linux-gnu.so

# Rule to build all files generated by this target.
CMakeFiles/mycpp.dir/build: mycpp.cpython-38-x86_64-linux-gnu.so

.PHONY : CMakeFiles/mycpp.dir/build

CMakeFiles/mycpp.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/mycpp.dir/cmake_clean.cmake
.PHONY : CMakeFiles/mycpp.dir/clean

CMakeFiles/mycpp.dir/depend:
	cd /home/apollolab/live-pose/FoundationPose/mycpp/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/apollolab/live-pose/FoundationPose/mycpp /home/apollolab/live-pose/FoundationPose/mycpp /home/apollolab/live-pose/FoundationPose/mycpp/build /home/apollolab/live-pose/FoundationPose/mycpp/build /home/apollolab/live-pose/FoundationPose/mycpp/build/CMakeFiles/mycpp.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/mycpp.dir/depend

