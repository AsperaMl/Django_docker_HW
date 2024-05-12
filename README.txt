# DJANGO BLOG Osinskiy A



cd .../dj_test/
docker-compose build
docker-compose up
docker-compose exec web python dj_test/manage.py makemigrations
docker-compose exec web python dj_test/manage.py migrate
docker-compose exec web python dj_test/manage.py makemigrations blog
docker-compose exec web python dj_test/manage.py migrate blog
