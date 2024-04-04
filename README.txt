# DJANGO BLOG

Диаграмма Ганта: https://docs.google.com/spreadsheets/d/1e2yOJ8LgywdKWXqPVtt3FK3DP-EmIEeqZr5zraWjIT0/edit?usp=sharing

cd .../dj_test/
docker-compose build
docker-compose up
docker-compose exec web python dj_test/manage.py makemigrations
docker-compose exec web python dj_test/manage.py migrate
docker-compose exec web python dj_test/manage.py makemigrations blog
docker-compose exec web python dj_test/manage.py migrate blog