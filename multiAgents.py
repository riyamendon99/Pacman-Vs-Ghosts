# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance
from game import Directions
import random, util
import sys
from game import Agent

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"
        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"

        # Setting the max possible integer value for the food distance
        min_distance_food = sys.maxsize

        # Converting the newFood Grid to a List for simplicity in computing
        newFood_list = newFood.asList()

        # Iterating through the food positions and finding the minimum distance between new position and food position   
        for food in newFood_list:
            food_distance = manhattanDistance(newPos, food)
            min_distance_food = min(min_distance_food, food_distance)

        newGhostPositions = successorGameState.getGhostPositions()

        # Iterating through ghost positions to find the distance between new position and ghost position
        for ghost in newGhostPositions:
            ghostDist = manhattanDistance(newPos, ghost)
            if(ghostDist < 2):
                return -sys.maxsize

        # If no food left in the grid, return negative infinity
        if not min_distance_food:
            return -sys.maxsize
        
        # Reciprocating the minimum distance as hinted in question so as to have maximum score for the nearest food
        updatedScore = successorGameState.getScore() + 1/min_distance_food
        return updatedScore

def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        return self.minimax(gameState, agentIndex=0, depth=self.depth)[1]

    # Defining a minimax function that takes the game state, agent index and depth as input arguments
    def minimax(self, gameState, agentIndex, depth):

        # Base condition for recursion. It returns the score when depth is 0 or game state is win or lose
        if depth == 0 or gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState), ""

        else:

            # Setting the initial action as null
            bestAction = ""

            # Decrementing the depth when agent index reaches the highest value so as to get the score at each depth
            if agentIndex == gameState.getNumAgents() - 1:
                depth = depth-1

            # Setting the initial best score value to lowest and highest numbers for agent 0 and other agents respectively    
            if agentIndex == 0:
                bestScore = -sys.maxsize
            else:
                bestScore = sys.maxsize

            # Iterating theough all the legal moves and recursively calculating the score for each agent at each depth
            legalMoves = gameState.getLegalActions(agentIndex)
            for action in legalMoves:
                successor = gameState.generateSuccessor(agentIndex, action)
                score = self.minimax(successor, (agentIndex+1)%gameState.getNumAgents(), depth)[0]

                # Since agent 0 is Pacman, it maximizes the utility and other agents minimize the utility
                if (agentIndex == 0 and score > bestScore) or (agentIndex != 0 and score < bestScore):
                    bestScore, bestAction = score, action

        # returning the best action an agent can take at a given depth
        return bestScore, bestAction 
        util.raiseNotDefined()

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"

        # Returning the best action for given game state
        return self.alphaBeta(gameState, agentIndex=0, depth=self.depth, alpha = -sys.maxsize, beta = sys.maxsize)[1]

    def alphaBeta(self, gameState, agentIndex, depth, alpha, beta):

        # Base condition for recursion is when we reach leaf state, we call the evaluation Function to return the score
        if depth == 0 or gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState), ""

        # Calculating the best score and best action for Pacman
        elif agentIndex == 0:

            # Reducing the depth to reach the leaf state
            if agentIndex == gameState.getNumAgents() - 1:
                depth = depth-1

            # Setting the values for best score and best action
            bestScore = -sys.maxsize
            bestAction = ""

            #Iterating through the legal actions and recursively calling the alphaBeta function to retrieve the score
            legalMoves = gameState.getLegalActions(agentIndex)
            for action in legalMoves:
                successor = gameState.generateSuccessor(agentIndex, action)
                score = self.alphaBeta(successor, (agentIndex +1)%gameState.getNumAgents(), depth, alpha, beta)[0]

                # For Pacman, update the score if it is greater than the present best score and update the action corresponding to best score
                if score > bestScore:
                    bestScore  = score
                    bestAction = action

                # If score is greater than beta, prune
                if score>beta:
                    return score, action

                # update the alpha value since we are considering max
                alpha = max(alpha, bestScore)
            return bestScore, bestAction
        else:
            
            if agentIndex == gameState.getNumAgents() - 1:
                depth = depth-1
            bestScore = sys.maxsize
            bestAction = ""
            legalMoves = gameState.getLegalActions(agentIndex)
            for action in legalMoves:
                successor = gameState.generateSuccessor(agentIndex, action)
                score = self.alphaBeta(successor, (agentIndex +1)%gameState.getNumAgents(), depth, alpha, beta)[0]

                # Calculating the best score and best action for ghosts
                # If the score is less than the current best score, update the best score and action
                if score < bestScore:
                    bestScore = score
                    bestAction = action

                # Prune the action if the score is less than the current alpha    
                if alpha > score:
                    return score, action

                # Update the beta value since we are considering min utility
                beta = min(beta, bestScore)
            return bestScore, bestAction

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        # Returning the best action possible for the expectimax agent
        return self.expectimax(gameState, agentIndex=0, depth=self.depth)[1]

    def expectimax(self, gameState, agentIndex, depth):
        # Base condition will call the evaluationFunction to return the score at dpeth 0
        if depth == 0 or gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState), ""
        else:

            # Setting score to negative infinity and initializing action to null
            bestScore = -sys.maxsize
            bestAction = ""

            # Decrementing the depth to reach leaf states
            if agentIndex == gameState.getNumAgents() - 1:
                depth = depth-1

            # Considering agent 0 or Pacman
            if agentIndex == 0:

                #Extracting all possible actions from a game state for a particular agent
                legalMoves = gameState.getLegalActions(agentIndex)

                #Iterating through the actions and calling the expectimax function recursively to return a score
                for action in legalMoves:
                    successor = gameState.generateSuccessor(agentIndex, action)
                    score = self.expectimax(successor, (agentIndex+1)%gameState.getNumAgents(), depth)[0]

                    # If the score is greater than the current best score, update score and action
                    # For Pacman, we need to maximize the utility
                    if score > bestScore:
                        bestScore = score
                        bestAction = action
                return bestScore, bestAction
            else:

                # Initializing the sum, average and best action for ghosts
                sumScore = 0
                average = 0
                bestAction = ""
                legalMoves = gameState.getLegalActions(agentIndex)
                for action in legalMoves:
                    successor = gameState.generateSuccessor(agentIndex, action)
                    score = self.expectimax(successor, (agentIndex+1)%gameState.getNumAgents(), depth)[0]

                    #Calculating the total of all scores
                    sumScore = sumScore + score

                # For ghosts, we need to calculate the average of all scores
                average = sumScore/len(legalMoves)
                return average, bestAction
                util.raiseNotDefined()

