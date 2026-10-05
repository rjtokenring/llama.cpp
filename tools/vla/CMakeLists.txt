# vla: vision-language-action policies

find_package(Threads REQUIRED)

add_library(vla
            vla.cpp
            vla.h
            vla-impl.h
            models/smolvla.cpp
            )

target_link_libraries     (vla PUBLIC ggml llama)
target_link_libraries     (vla PRIVATE Threads::Threads)
target_include_directories(vla PUBLIC .)
target_compile_features   (vla PRIVATE cxx_std_17)

if (BUILD_SHARED_LIBS)
    set_target_properties     (vla PROPERTIES POSITION_INDEPENDENT_CODE ON)
    target_compile_definitions(vla PRIVATE LLAMA_BUILD)
    target_compile_definitions(vla PUBLIC  LLAMA_SHARED)
endif()

set_target_properties(vla PROPERTIES PUBLIC_HEADER ${CMAKE_CURRENT_SOURCE_DIR}/vla.h)
install(TARGETS vla LIBRARY PUBLIC_HEADER)

set(TARGET llama-vla-cli)
add_executable(${TARGET} vla-cli.cpp)
target_link_libraries(${TARGET} PRIVATE vla llama vendor::stb)
target_compile_features(${TARGET} PRIVATE cxx_std_17)

if (NOT MSVC)
    # for stb_image.h
    target_compile_options(${TARGET} PRIVATE -Wno-cast-qual)
endif()

if(LLAMA_TOOLS_INSTALL)
    install(TARGETS ${TARGET} RUNTIME)
endif()

set(TARGET llama-vla-server)
add_executable(${TARGET} vla-server.cpp)
target_link_libraries(${TARGET} PRIVATE vla llama llama-common cpp-httplib vendor::stb ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(${TARGET} PRIVATE cxx_std_17)

if (NOT MSVC)
    # for stb_image.h
    target_compile_options(${TARGET} PRIVATE -Wno-cast-qual)
endif()

if(LLAMA_TOOLS_INSTALL)
    install(TARGETS ${TARGET} RUNTIME)
endif()
