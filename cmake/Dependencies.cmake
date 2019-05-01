
# ---[ Anti-Grain Geometry
# (used in data-generating layer)
ExternalProject_Add(
        AntiGrainGeometry
        PREFIX "${CMAKE_BINARY_DIR}/AntiGrainGeometry"
        URL "http://www.antigrain.com/agg-2.4.tar.gz"
        URL_HASH "MD5=863d9992fd83c5d40fe1c011501ecf0e"
        # do not update
        UPDATE_COMMAND ""
        PATCH_COMMAND patch src/Makefile < ${PROJECT_SOURCE_DIR}/src/AntiGrainGeometry-Makefile-patch-fPIC.txt COMMAND patch include/agg_scanline_u.h < ${PROJECT_SOURCE_DIR}/src/AntiGrainGeometry-agg_scanline_u-patch.txt
        # do not configure
        CONFIGURE_COMMAND ""
        BUILD_IN_SOURCE 1
        # compile with PIC because caffe builds a shared object
        BUILD_COMMAND make
        # do not install
        INSTALL_COMMAND ""
)
ExternalProject_Get_Property( AntiGrainGeometry SOURCE_DIR BINARY_DIR )
include_directories( "${SOURCE_DIR}/include" )
list(APPEND Caffe_LINKER_LIBS ${SOURCE_DIR}/src/libagg.a )

