project (darknetp C)

set(THREADS_PREFER_PTHREAD_FLAG ON)

set( THREADS_PTHREAD_ARG 
     "0"
     CACHE STRING "Result from TRY_RUN" FORCE)

find_package(Threads REQUIRED)

FILE(GLOB SRC host/src/*.c)
FILE(GLOB EXAM host/examples/*.c)
set(MAIN host/main.c)

add_executable (${PROJECT_NAME} ${SRC} ${EXAM} ${MAIN})

target_include_directories(${PROJECT_NAME}
				 PRIVATE host/include
				 PRIVATE host/src
				 PRIVATE include)

target_link_libraries (${PROJECT_NAME} PRIVATE teec -lm Threads::Threads)

install (TARGETS ${PROJECT_NAME} DESTINATION ${CMAKE_INSTALL_BINDIR})
