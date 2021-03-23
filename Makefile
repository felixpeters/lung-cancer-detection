build-image:
	docker build -t felixpeters/lung-cancer-detection . -f docker/Dockerfile

run-image:
	docker run --name=lct-dev -it -v $(shell pwd):/code -v /Volumes/LaCie/data/lung-cancer-detection/lidc-idri:/data -p 8080:8080 felixpeters/lung-cancer-detection

upload-image:
	docker push felixpeters/lung-cancer-detection:latest