# CMAKE generated file: DO NOT EDIT!
# Generated by "MinGW Makefiles" Generator, CMake Version 3.20

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

SHELL = cmd.exe

# The CMake executable.
CMAKE_COMMAND = D:\CMaKe\bin\cmake.exe

# The command to remove a file.
RM = D:\CMaKe\bin\cmake.exe -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = C:\Users\12235\Desktop\prove

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = C:\Users\12235\Desktop\prove\build

# Include any dependencies generated for this target.
include CMakeFiles/detect_test.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/detect_test.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/detect_test.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/detect_test.dir/flags.make

CMakeFiles/detect_test.dir/main.cpp.obj: CMakeFiles/detect_test.dir/flags.make
CMakeFiles/detect_test.dir/main.cpp.obj: CMakeFiles/detect_test.dir/includes_CXX.rsp
CMakeFiles/detect_test.dir/main.cpp.obj: ../main.cpp
CMakeFiles/detect_test.dir/main.cpp.obj: CMakeFiles/detect_test.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=C:\Users\12235\Desktop\prove\build\CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/detect_test.dir/main.cpp.obj"
	D:\TDM-GCC-64\bin\x86_64-w64-mingw32-g++.exe $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/detect_test.dir/main.cpp.obj -MF CMakeFiles\detect_test.dir\main.cpp.obj.d -o CMakeFiles\detect_test.dir\main.cpp.obj -c C:\Users\12235\Desktop\prove\main.cpp

CMakeFiles/detect_test.dir/main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/detect_test.dir/main.cpp.i"
	D:\TDM-GCC-64\bin\x86_64-w64-mingw32-g++.exe $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E C:\Users\12235\Desktop\prove\main.cpp > CMakeFiles\detect_test.dir\main.cpp.i

CMakeFiles/detect_test.dir/main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/detect_test.dir/main.cpp.s"
	D:\TDM-GCC-64\bin\x86_64-w64-mingw32-g++.exe $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S C:\Users\12235\Desktop\prove\main.cpp -o CMakeFiles\detect_test.dir\main.cpp.s

# Object files for target detect_test
detect_test_OBJECTS = \
"CMakeFiles/detect_test.dir/main.cpp.obj"

# External object files for target detect_test
detect_test_EXTERNAL_OBJECTS =

detect_test.exe: CMakeFiles/detect_test.dir/main.cpp.obj
detect_test.exe: CMakeFiles/detect_test.dir/build.make
detect_test.exe: libdetector.dll.a
detect_test.exe: D:/Program\ Files/Intel/openvino_2021.4.752/opencv/lib/opencv_gapi453d.lib
detect_test.exe: D:/Program\ Files/Intel/openvino_2021.4.752/opencv/lib/opencv_highgui453d.lib
detect_test.exe: D:/Program\ Files/Intel/openvino_2021.4.752/opencv/lib/opencv_ml453d.lib
detect_test.exe: D:/Program\ Files/Intel/openvino_2021.4.752/opencv/lib/opencv_objdetect453d.lib
detect_test.exe: D:/Program\ Files/Intel/openvino_2021.4.752/opencv/lib/opencv_photo453d.lib
detect_test.exe: D:/Program\ Files/Intel/openvino_2021.4.752/opencv/lib/opencv_stitching453d.lib
detect_test.exe: D:/Program\ Files/Intel/openvino_2021.4.752/opencv/lib/opencv_video453d.lib
detect_test.exe: D:/Program\ Files/Intel/openvino_2021.4.752/opencv/lib/opencv_calib3d453d.lib
detect_test.exe: D:/Program\ Files/Intel/openvino_2021.4.752/opencv/lib/opencv_dnn453d.lib
detect_test.exe: D:/Program\ Files/Intel/openvino_2021.4.752/opencv/lib/opencv_features2d453d.lib
detect_test.exe: D:/Program\ Files/Intel/openvino_2021.4.752/opencv/lib/opencv_flann453d.lib
detect_test.exe: D:/Program\ Files/Intel/openvino_2021.4.752/opencv/lib/opencv_videoio453d.lib
detect_test.exe: D:/Program\ Files/Intel/openvino_2021.4.752/opencv/lib/opencv_imgcodecs453d.lib
detect_test.exe: D:/Program\ Files/Intel/openvino_2021.4.752/opencv/lib/opencv_imgproc453d.lib
detect_test.exe: D:/Program\ Files/Intel/openvino_2021.4.752/opencv/lib/opencv_core453d.lib
detect_test.exe: D:/Program\ Files/Intel/openvino_2021.4.752/deployment_tools/inference_engine/lib/intel64/Debug/inference_engined.lib
detect_test.exe: D:/Program\ Files/Intel/openvino_2021.4.752/deployment_tools/inference_engine/lib/intel64/Debug/inference_engine_c_apid.lib
detect_test.exe: D:/Program\ Files/Intel/openvino_2021.4.752/deployment_tools/ngraph/lib/ngraph.dll
detect_test.exe: CMakeFiles/detect_test.dir/linklibs.rsp
detect_test.exe: CMakeFiles/detect_test.dir/objects1.rsp
detect_test.exe: CMakeFiles/detect_test.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=C:\Users\12235\Desktop\prove\build\CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable detect_test.exe"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles\detect_test.dir\link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/detect_test.dir/build: detect_test.exe
.PHONY : CMakeFiles/detect_test.dir/build

CMakeFiles/detect_test.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles\detect_test.dir\cmake_clean.cmake
.PHONY : CMakeFiles/detect_test.dir/clean

CMakeFiles/detect_test.dir/depend:
	$(CMAKE_COMMAND) -E cmake_depends "MinGW Makefiles" C:\Users\12235\Desktop\prove C:\Users\12235\Desktop\prove C:\Users\12235\Desktop\prove\build C:\Users\12235\Desktop\prove\build C:\Users\12235\Desktop\prove\build\CMakeFiles\detect_test.dir\DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/detect_test.dir/depend

