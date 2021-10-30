

class myMl():
    def __init__(self, name):
        self.Name = name
    
    def Greeting(self, message):
        self.Message = message
    
    def Introduce(self):
        print(f"Hello, my name is {self.Name}, {self.Message}")

# init Person
dude = Person("Karim")

# declare greeting
dude.Greeting("nice to meet you")

dude.Introduce()

