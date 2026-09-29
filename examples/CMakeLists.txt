add_subdirectory(1-how-to-use)
add_subdirectory(2-linear-vision-pipline)
add_subdirectory(3-joined-vision-pipline)
if(WITH_AIFOOTBALL_SDK)
  # The touch business logic lives in Python (examples/touch_eval/py), so
  # the app only runs the models and writes observations.jsonl.
  add_subdirectory(aifootball_app)
endif()
