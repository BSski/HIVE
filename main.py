import random
iterations = 40

def generate_plan(iterations):
    plan = ""
    for i in range(0, iterations):
        plan += random.choice("01")
    print(plan)
    return plan

plan0 = generate_plan(iterations)
plan1 = generate_plan(iterations)
#plan0 = "00000000000000000000"
#plan1 = "11111111111111111111"


class Agent:
    def __init__(self, plan, cash = 500):
        self.cash = cash
        self.plan = plan


    def get_cash(self):
        return self.cash


    def change_cash(self, value):
        self.cash += value


    def move(self, plan, i):
        return int(plan[i])  #int(random.choice("01"))


    def get_plan(self):
        return self.plan


    def play(self, player2, iterations):
        game = []
        for i in range(0, iterations):
            self.change_cash(-10)
            player2.change_cash(-10)
            agent0_move = self.move(self.get_plan(), i)
            agent1_move = player2.move(player2.get_plan(), i)
            if agent0_move == 0 and agent1_move == 0:
                pass
            if agent0_move == 1 and agent1_move == 0:
                self.change_cash(25)
                player2.change_cash(-10)
            if agent0_move == 0 and agent1_move == 1:
                self.change_cash(-10)
                player2.change_cash(25)
            if agent0_move == 1 and agent1_move == 1:
                self.change_cash(15)
                player2.change_cash(15)
            game.append([agent0_move, agent1_move])
            print(self.get_cash(), player2.get_cash(), agent0_move, agent1_move)
        print(game)


# Logic
agent0 = Agent(plan0)
agent1 = Agent(plan1)

agent0.play(agent1, iterations)
