build:
	docker build -t felixpeters/lung-cancer-detection . -f docker/Dockerfile

run:
	docker run -it -v $(shell pwd):/code -v /Volumes/LaCie/data/lung-cancer-detection/lidc-idri/processed:/data -p 8080:8080 felixpeters/lung-cancer-detection

publish:
	docker push felixpeters/lung-cancer-detection:latest

test:
	rm -R data/test/cache && pytest --disable-pytest-warnings
