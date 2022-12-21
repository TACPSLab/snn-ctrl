devel:
	make -C ./deeprl install
	catkin_make --directory ../ -DPYTHON_EXECUTABLE=/usr/bin/python3.8 -DCMAKE_EXPORT_COMPILE_COMMANDS=1

clean:
	rm -rf ../build
	rm -rf ../devel
	rm -rf ../install
