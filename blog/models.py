from django.db import models as ml


class Post(ml.Model):
    title = ml.CharField(max_length=500)
    author = ml.ForeignKey(
        'auth.User',
        on_delete=ml.CASCADE
    )
    body = ml.TextField()

    def __str__(self):
        return self.title


class DbPolRegression(ml.Model):
    year = ml.IntegerField()
    month = ml.CharField(max_length=30)
    passengers = ml.FloatField()
    predictions = ml.FloatField(null=True)
    wrong_answer = ml.FloatField(null=True)

    def __int__(self):
        return self.year

    def __float__(self):
        return self.passengers, self.predictions, self.wrong_answer

class DbGradient(ml.Model):
    age = ml.IntegerField()
    gender = ml.CharField(max_length=30)
    platform = ml.CharField(max_length=30)
    y_pred = ml.FloatField()

    def __int__(self):
        return self.age

    def __float__(self):
        return self.y_pred
class DbRNN(ml.Model):
    y_pred = ml.FloatField()



