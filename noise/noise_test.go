package noise

import (
	"math/rand"
	"testing"
)

func TestNoise(t *testing.T) {
	seed := int64(10)
	size := 10
	chunkSize := 5
	r := rand.New(rand.NewSource(seed))
	table := New(seed, size)

	fullChunk := table.Chunk(0, chunkSize)
	randIndex := table.SampleIndex(r, chunkSize)
	randChunk := table.Chunk(0, randIndex+1)

	if !fullChunk.Overlaps(randChunk) {
		t.Errorf("Chunks should overlap!")
	}
}
