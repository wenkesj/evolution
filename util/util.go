// Package util implements utility functions and data structures
//
// For argsort adaptation:
//
// Copyright 2013 The Gonum Authors. All rights reserved.
// Use of this code is governed by a BSD-style
// license that can be found in the LICENSE file
package util

import (
	"sort"
)

// argsort is a helper that implements sort.Interface, as used by
// Argsort.
type argsort struct {
	s    []float64
	Inds []int
}

func (a argsort) Len() int {
	return len(a.s)
}

func (a argsort) Less(i, j int) bool {
	return a.s[i] < a.s[j]
}

func (a argsort) Swap(i, j int) {
	a.s[i], a.s[j] = a.s[j], a.s[i]
	a.Inds[i], a.Inds[j] = a.Inds[j], a.Inds[i]
}

// Argsort sorts the src and returns the indices of the sorted array.
func Argsort(src []float64) []int {
	inds := make([]int, len(src))
	for i := range src {
		inds[i] = i
	}

	a := argsort{s: src, Inds: inds}
	sort.Sort(a)
	return a.Inds
}

// Ravel flattens the 2-d array into 1-d.
func Ravel(x [][2]float64) []float64 {
	var ret []float64
	for _, y := range x {
		ret = append(ret, y[0], y[1])
	}
	return ret
}

// ComputeCenteredRank computes the ranks adjusted to be centered.
func ComputeCenteredRank(x [][2]float64) [][2]float64 {
	y := Ravel(x)
	ranks := make([]int, len(y))
	argSorted := Argsort(y)
	for i := range ranks {
		ranks[argSorted[i]] = i
	}

	// Reshape
	k := 0
	centeredRanks := make([][2]float64, len(x))
	for i := range x {
		centeredRanks[i] = [2]float64{0, 0}
		for j := range x[i] {
			centeredRanks[i][j] = float64(ranks[k] / (len(x) - 1))
			centeredRanks[i][j] -= 0.5
			k++
		}
	}
	return centeredRanks
}
