devel:
	catkin_make --directory ../ -DPYTHON_EXECUTABLE=/usr/bin/python3.8 -DCMAKE_EXPORT_COMPILE_COMMANDS=1

all: devel
	make -C ./deeprl install

clean:
	rm -rf ../build
	rm -rf ../devel
	rm -rf ../install
