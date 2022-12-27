devel:
	catkin_make --directory ../ -DPYTHON_EXECUTABLE=/usr/bin/python3.8 -DCMAKE_EXPORT_COMPILE_COMMANDS=1

all: devel
	python3 -m pip install ./deeprl
	make -C ./gym_exps all-rqmts

clean:
	rm -rf ../build
	rm -rf ../devel
	rm -rf ../install
