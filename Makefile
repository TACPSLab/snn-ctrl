build:
	cd ..; colcon build --symlink-install

all: build
	make -C ./deeprl install

clean:
	rm -rf ../build
	rm -rf ../install
