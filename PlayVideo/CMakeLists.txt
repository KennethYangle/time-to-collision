project( PlayVideo )
find_package( OpenCV REQUIRED )
add_executable( playvideo play.cpp )
target_link_libraries( playvideo ${OpenCV_LIBS} )

