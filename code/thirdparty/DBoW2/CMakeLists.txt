cmake_minimum_required(VERSION 2.8)
project(DBoW2)

set(LOCAL_INSTALL_DIR /home/paleshnikov/install/)  # adjust this variable to match the local installation directory of the OpenCV library

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}  -Wall  -O3 -march=native ")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall  -O3 -march=native")

set(HDRS_DBOW2
  DBoW2/BowVector.h
  DBoW2/FORB.h 
  DBoW2/FClass.h       
  DBoW2/FeatureVector.h
  DBoW2/ScoringObject.h
  DBoW2/QueryResults.h   
  DBoW2/TemplatedVocabulary.h
  DBoW2/TemplatedDatabase.h)
set(SRCS_DBOW2
  DBoW2/BowVector.cpp
  DBoW2/FORB.cpp      
  DBoW2/FeatureVector.cpp
  DBoW2/QueryResults.cpp   
  DBoW2/ScoringObject.cpp)

set(HDRS_DVISION DVision/FSolver.h)

set(SRCS_DVISION DVision/FSolver.cpp)

set(HDRS_DUTILS
  DUtils/Random.h
  DUtils/Timestamp.h)
set(SRCS_DUTILS
  DUtils/Random.cpp
  DUtils/Timestamp.cpp)

find_package(OpenCV REQUIRED PATHS ${LOCAL_INSTALL_DIR}/local/share/OpenCV NO_DEFAULT_PATH)

set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib)

include_directories(${OpenCV_INCLUDE_DIRS})
add_library(DBoW2 SHARED ${SRCS_DBOW2} ${SRCS_DUTILS} ${SRCS_DVISION})
target_link_libraries(DBoW2 ${OpenCV_LIBS})

