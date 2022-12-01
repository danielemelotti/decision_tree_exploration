# Plotting the data to inspect relationships
plot(fulldata[, c(dv, "TOT_LVG_AREA", "RAIL_DIST", "CNTR_DIST", "age", "structure_quality")], col=rgb(0.7, 0.7, 0.7, 0.3)) # inspecting the relationships between the data

# Visualizing the tree
rpart.plot(tree, type = 2)

# DESCRIPTION: The top value is the average house price for the houses included within a certain node. The lower value represents the percentage of the whole data which is included at that node (and therefore satisfies the decision criteria at every split up to that point).

# COMMENT: In classification trees there is one more value in each node, just between the two just described values, which is the probability. See https://www.guru99.com/r-decision-trees.html for an extensive explanation of classification trees.

# Visualizing the pruned tree
rpart.plot(prune_tree)

## Showing the set of possible cost-complexity prunings of a tree from a nested set with plotcp().
# https://rdrr.io/cran/itree/man/plotcp.html
# We can compare the tree before vs after pruning
plotcp(tree)
plotcp(prune_tree)

# DESCRIPTION: What is the dotted line in plotcp()? The function plots the cp values against their related xerror, with st dev included.The dotted line represents the highest cross-validated error less than the minimum cross-validated error plus 1 standard deviation of the error at that tree. See https://stackoverflow.com/questions/21698540/whats-the-meaning-of-plotcp-result-in-rpart for reference.

# Visualizing a tree with cp = 0.007
rpart.plot(tree_fun, type = 2)

