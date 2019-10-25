#!/bin/sh

i=1

echo "Test $i"
./bandit.sh --instance ../instances/i-1.txt --algorithm round-robin --randomSeed 40 --horizon 200 --epsilon 1
sleep 1
i=$((i + 1))

echo "Test $i"
./bandit.sh --instance ../instances/i-1.txt --algorithm epsilon-greedy --epsilon 0.02 --horizon 800 --randomSeed 10 
sleep 1
i=$((i + 1))

echo "Test $i"
./bandit.sh --instance ../instances/i-2.txt --algorithm epsilon-greedy --epsilon 0.002 --randomSeed 11 --horizon 12800
sleep 1
i=$((i + 1))

echo "Test $i"
./bandit.sh --instance ../instances/i-3.txt --algorithm epsilon-greedy --epsilon 0.2 --randomSeed 1 --horizon 3200
sleep 1
i=$((i + 1))

echo "Test $i"
./bandit.sh --instance ../instances/i-1.txt --algorithm ucb --randomSeed 2 --horizon 200 --epsilon 1
sleep 1
i=$((i + 1))

echo "Test $i"
./bandit.sh --instance ../instances/i-3.txt --algorithm ucb --randomSeed 22 --horizon 12800 --epsilon 1 
sleep 1 
i=$((i + 1))

echo "Test $i"
./bandit.sh --instance ../instances/i-2.txt --algorithm kl-ucb --randomSeed 25 --horizon 800 --epsilon 1
sleep 1
i=$((i + 1))

echo "Test $i"
./bandit.sh --instance ../instances/i-1.txt --algorithm kl-ucb --randomSeed 25 --horizon 50 --epsilon 1
sleep 1
i=$((i + 1))

echo "Test $i"
./bandit.sh --instance ../instances/i-2.txt --algorithm thompson-sampling --randomSeed 15 --horizon 200 --epsilon 1
sleep 1
i=$((i + 1))




