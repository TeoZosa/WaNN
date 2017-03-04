
def update_tree_wins(node, amount=1): #visits and wins go together
    node.wins += amount
    node.visits += amount
    parent = node.parent
    while parent != None:
        parent.wins += amount
        parent.visits += amount
        parent = parent.parent

def update_tree_visits(node, amount=1):
    node.visits += amount
    parent = node.parent
    while parent != None:
        parent.visits += amount
        parent = parent.parent