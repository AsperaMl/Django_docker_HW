# DJANGO BLOG Osinskiy A

Диаграмма Ганта: https://docs.google.com/spreadsheets/d/1ynuhvYqWto7j_e877yzo4kw1EmutcgMrxKRelrU8oNw/edit?usp=sharing

cd .../dj_test/
docker-compose build
docker-compose up
docker-compose exec web python dj_test/manage.py makemigrations
docker-compose exec web python dj_test/manage.py migrate
docker-compose exec web python dj_test/manage.py makemigrations blog
docker-compose exec web python dj_test/manage.py migrate blog
