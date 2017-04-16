// Package noise implements noise sharing and Gaussian binning
package noise

import (
	"github.com/unixpickle/anyvec"
	"github.com/unixpickle/anyvec/anyvec64"
	"math/rand"
)

// NoiseTable is a slice of Gaussian noise.
type NoiseTable struct {
	noise anyvec.Vector
}

// New creates a new NoiseTable from a seed, sampled from a standard normal
// distribution.
func New(seed int64, size int) *NoiseTable {
	r := rand.New(rand.NewSource(seed))
	table := make([]float64, size)
	for i := 0; i < size; i++ {
		table[i] = r.NormFloat64()
	}
	return &NoiseTable{
		noise: anyvec64.MakeVectorData(anyvec64.MakeNumericList(table)),
	}
}

// Chunk returns a chunk of noise from start to start + end.
func (table *NoiseTable) Chunk(start, end int) anyvec.Vector {
	return table.noise.Slice(start, start+end)
}

// SampleIndex returns an integer, sampled from the given source, representing
// an index from the noise table.
func (table *NoiseTable) SampleIndex(source *rand.Rand, dim int) int {
	return source.Intn(table.noise.Len() - dim + 1)
}
