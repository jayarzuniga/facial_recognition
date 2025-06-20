from django.db import models

class UserFace(models.Model):
    name = models.CharField(max_length=100)
    embedding = models.BinaryField()  

    def __str__(self):
        return self.name
