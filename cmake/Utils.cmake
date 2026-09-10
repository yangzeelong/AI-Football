# 把列表参数直接当作列表处理
function(separate_files_by_suffix SUFFIX INPUT_LIST OUT_REGULAR_FILES OUT_MATCHED_FILES)
    # 本地结果列表
    set(local_regular)
    set(local_matched)

    # 把 "." 转义为 "\."，便于正则匹配
    string(REPLACE "." "\\." escaped_suffix "${SUFFIX}")

    # 这里直接用变量 INPUT_LIST（它已包含列表内容）
    foreach(current_file IN LISTS INPUT_LIST)
        if("${current_file}" MATCHES "${escaped_suffix}$")
            list(APPEND local_matched "${current_file}")
        else()
            list(APPEND local_regular "${current_file}")
        endif()
    endforeach()

    # 返回到父作用域
    set(${OUT_REGULAR_FILES} ${local_regular} PARENT_SCOPE)
    set(${OUT_MATCHED_FILES} ${local_matched} PARENT_SCOPE)
endfunction()
