#!/usr/bin/env python
# coding: utf-8

# # COMP4660/8420 Fuzzy Logic

# ## Part 2: Introduction to scikit-fuzzy

# Now it’s time to apply the concepts to some examples using a Python package called [scikit-fuzzy](http://pythonhosted.org/scikit-fuzzy/). Working through these
# practical examples will help to develop your understanding.

# ### Task 1: Fuzzy Inference System for a Temperature Control System

# *Goal: *Understand and implement a basic inference system in scikit-fuzzy.

# Your first task is to build a Mamdani style inference system. You may find [this link](http://pythonhosted.org/scikit-fuzzy/auto_examples/plot_tipping_problem.html) to be helpful for you understanding scikit-fuzzy. To get started, please run 

# `pip install scikit-fuzzy`

# to install scikit-fuzzy on your machine. 

# We will work through another example which is a basic temperature control system that
# was discussed in the lectures

# ### 1. Import all the packages that you will need during this lab.

# In[1]:


# ==================================
# A Temperature Control System - The Hard Way
# ==================================

#  Note: This method computes everything by hand, step by step. For most people,
#  the new API for fuzzy systems will be preferable. The same problem is solved
#  with the new API `in this example <./plot_tipping_problem_newapi.html>`_.

# Input variables
# ---------------

# A number of variables play into the decision about how much power while
# dining. Consider one item:

# * ``temp`` : Temperature

# Output variable
# ---------------

# The output variable is simply the power, in percentage points:

# * ``power`` : Percent of power


# For the purposes of discussion, let's say we need 'cold', 'pleasant', and 'hot' 
# for input variables as well as 'high', 'medium', and 'low' membership functions 
# for output variable. These are defined in scikit-fuzzy as follows


# In[2]:


import numpy as np
import skfuzzy as fuzz
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# ### 2. Generate universal variables 

# In[3]:


# Generate universe variables
#   * Temperature ranges [0, 45]
#   * Power has a range of [0, 100]

x_temp = np.arange(0, 45, 1)
x_power = np.arange(0, 100, 1)


# ### 3. Generate fuzzy membership functions

# In[4]:


# Generate fuzzy membership functions
temp_cold = fuzz.trimf(x_temp, [0, 8, 23])
temp_pleasant = fuzz.trimf(x_temp, [6, 20, 30])
temp_hot = fuzz.trimf(x_temp, [23, 35, 45])

power_low = fuzz.trimf(x_power, [0, 15, 30])
power_mid = fuzz.trimf(x_power, [30, 45, 60])
power_high = fuzz.trimf(x_power, [60, 75, 100])


# In[5]:


# This is for visualization purposes.
power0 = np.zeros_like(x_power)


# In[6]:


# Visualize these universes and membership functions
fig, (ax0, ax1) = plt.subplots(nrows=2, figsize=(8, 6))

ax0.plot(x_temp, temp_cold, 'b', linewidth=1.5, label='Cold')
ax0.plot(x_temp, temp_pleasant, 'g', linewidth=1.5, label='Pleasant')
ax0.plot(x_temp, temp_hot, 'r', linewidth=1.5, label='Hot')
ax0.set_title('Temperature')
ax0.legend()

ax1.plot(x_power, power_low, 'b', linewidth=1.5, label='Low')
ax1.plot(x_power, power_mid, 'g', linewidth=1.5, label='Mid')
ax1.plot(x_power, power_high, 'r', linewidth=1.5, label='High')
ax1.set_title('Power')
ax1.legend()

for ax in (ax0, ax1):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()

plt.tight_layout()


# In[7]:


# Fuzzy rules
# -----------

# Now, to make these triangles useful, we define the *fuzzy relationship*
# between input and output variables. For the purposes of our example, consider
# three simple rules:

# 1. If (Temp is Cold) then (Power is Low)
# 2. If (Temp is Pleasant) then (Power is Med)
# 3. If (Temp is Hot) then (Power is High)

# Most people would agree on these rules, but the rules are fuzzy. Mapping the
# imprecise rules into a defined, actionable power is a challenge. This is the
# kind of task at which fuzzy logic excels.


# In[8]:


def apply_rule_to_get_activation_values(temp, visualizing=False):
    temp_level_cold = fuzz.interp_membership(x_temp, temp_cold, temp)
    temp_level_pleasant = fuzz.interp_membership(x_temp, temp_pleasant, temp)
    temp_level_hot = fuzz.interp_membership(x_temp, temp_hot, temp)

    # Now we apply this by clipping the top off the corresponding output
    # membership function with `np.fmin`
    power_activation_low = np.fmin(temp_level_cold, power_low)
    power_activation_mid = np.fmin(temp_level_pleasant, power_mid)
    power_activation_high = np.fmin(temp_level_hot, power_high)

    # Visualize this
    if visualizing:
        fig, ax0 = plt.subplots(figsize=(8, 3))

        ax0.fill_between(x_power, power0, power_activation_low, facecolor='b', alpha=0.7)
        ax0.plot(x_power, power_low, 'b', linewidth=0.5, linestyle='--', )
        ax0.fill_between(x_power, power0, power_activation_mid, facecolor='g', alpha=0.7)
        ax0.plot(x_power, power_mid, 'g', linewidth=0.5, linestyle='--')
        ax0.fill_between(x_power, power0, power_activation_high, facecolor='r', alpha=0.7)
        ax0.plot(x_power, power_high, 'r', linewidth=0.5, linestyle='--')
        ax0.set_title('Output membership activity')

        # Turn off top/right axes
        for ax in (ax0,):
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.get_xaxis().tick_bottom()
            ax.get_yaxis().tick_left()

        plt.tight_layout()
        plt.show()

    return power_activation_low, power_activation_mid, power_activation_high

fuzzifications = apply_rule_to_get_activation_values(30, True)


# In[9]:


# Defuzzification
# ---------------
# Finally, to get a real world answer, we return to *crisp* logic from the
# world of fuzzy membership functions. For the purposes of this example
# the centroid method will be used.


# In[10]:


def defuzzify(fuzzifications, visualizing=False):
    # Aggregate all three output membership functions together
    aggregated = np.fmax(fuzzifications[0],
                         np.fmax(fuzzifications[1], fuzzifications[2]))

    # Calculate defuzzified result
    power = fuzz.defuzz(x_power, aggregated, 'centroid')
    power_activation = fuzz.interp_membership(x_power, aggregated, power)  # for plot

    # Visualize this
    if visualizing:
        fig, ax0 = plt.subplots(figsize=(8, 3))

        ax0.plot(x_power, power_low, 'b', linewidth=0.5, linestyle='--', )
        ax0.plot(x_power, power_mid, 'g', linewidth=0.5, linestyle='--')
        ax0.plot(x_power, power_high, 'r', linewidth=0.5, linestyle='--')
        ax0.fill_between(x_power, power0, aggregated, facecolor='Orange', alpha=0.7)
        ax0.plot([power, power], [0, power_activation], 'k', linewidth=1.5, alpha=0.9)
        ax0.set_title('Aggregated membership and result (line)')

        # Turn off top/right axes
        for ax in (ax0,):
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.get_xaxis().tick_bottom()
            ax.get_yaxis().tick_left()

        plt.tight_layout()
        plt.show()

defuzzify(fuzzifications, True)


# #### Q1. What are the linguistic variables and the universe of discourse of each variable?
# 
# #### Q2. What are the fuzzy sets?
# 
# #### Q3. What are the fuzzy rules?
# 
# #### Q4. What is the defuzzification method?
# 
# Refer back to the lecture notes on Fuzzy Rule Based Systems to read through the Air
# Conditioner example. Change the fuzzy sets for the Temperature variable to be the
# same as in the lecture notes. Then change the fuzzy sets for the Power variable to be
# the same as the Speed Fuzzy sets in the lectures. Now modify the fuzzy rules to be
# the same as in the lecture notes. Now evaluate the FIS with the same inputs used in
# step 4. 
# 
# #### Q5. What results do you get now? Do you think the modified FIS more accurately portrays the problem space? Why?
# 
# #### Q6. Do you agree with the range of the universe of discourse and the linguistic variableused? Are there any factors that you think have been overlooked and would be useful in altering the speed of the air conditioner?
# 
# Play around with FIS settings and comment on how it changes the FIS mode. You
# may like to experiment with different membership functions, more variables,
# different defuzzification methods etc. 

