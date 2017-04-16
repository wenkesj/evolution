// Package env implements the gym-http-api utilites.
package env

import (
	gym "github.com/openai/gym-http-api/binding-go"
)

// New creates a new client from the given url and constructs an environment
func New(url, environment string) (*gym.Client, gym.InstanceID, error) {
	client, err := gym.NewClient(url)
	if err != nil {
		return client, gym.InstanceID(""), err
	}

	// Create environment instance.
	id, err := client.Create(environment)
	return client, id, err
}
