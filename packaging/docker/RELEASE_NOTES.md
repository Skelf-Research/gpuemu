# Docker release workflow (manual addition required)

The `Dockerfile` in the repo root and the `integrations/github-action/` PoC
ship in this branch, but the `.github/workflows/release.yml` modification that
publishes the image to ghcr.io was **not** included because the bot pushing
this branch lacks the GitHub `workflow` OAuth scope.

To complete the Docker publish path, append the snippet below to
`.github/workflows/release.yml` in a separate PR pushed from a workstation
with `workflow` scope (or via the GitHub web UI editor).

## What to add

In `permissions:`, alongside `contents: write`, add:

```yaml
permissions:
  contents: write
  packages: write    # <-- new, needed for ghcr.io push
```

After the existing `release:` job, append:

```yaml
  docker:
    needs: build
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v4

    - name: Get version
      id: version
      run: |
        if [ -n "${{ github.event.inputs.tag }}" ]; then
          echo "version=${{ github.event.inputs.tag }}" >> $GITHUB_OUTPUT
        else
          echo "version=${GITHUB_REF#refs/tags/}" >> $GITHUB_OUTPUT
        fi

    - name: Log in to ghcr.io
      uses: docker/login-action@v3
      with:
        registry: ghcr.io
        username: ${{ github.actor }}
        password: ${{ secrets.GITHUB_TOKEN }}

    - name: Set up Docker Buildx
      uses: docker/setup-buildx-action@v3

    # Two images: the lean default (no torch), and the CI variant the
    # gpuemu/validate-action consumes (WITH_TORCH=1). Both go under the
    # GitHub Container Registry namespace `ghcr.io/<owner>/gpuemu`.

    - name: Build and push (lean)
      uses: docker/build-push-action@v6
      with:
        context: .
        push: true
        platforms: linux/amd64,linux/arm64
        tags: |
          ghcr.io/${{ github.repository_owner }}/gpuemu:${{ steps.version.outputs.version }}
          ghcr.io/${{ github.repository_owner }}/gpuemu:latest

    - name: Build and push (with-torch CI variant)
      uses: docker/build-push-action@v6
      with:
        context: .
        push: true
        platforms: linux/amd64
        build-args: |
          WITH_TORCH=1
        tags: |
          ghcr.io/${{ github.repository_owner }}/gpuemu:${{ steps.version.outputs.version }}-with-torch
          ghcr.io/${{ github.repository_owner }}/gpuemu:ci
```

Also extend the release-notes body string in the existing `release:` job to
mention the docker tag (optional but informative):

```yaml
          ### Docker
          ```bash
          docker run --rm ghcr.io/${{ github.repository_owner }}/gpuemu:${{ steps.version.outputs.version }} version
          ```
```

## Why this is split off

The OAuth flow that authenticates this branch's pusher granted scopes
`gist`, `read:org`, `repo` but not `workflow`. GitHub blocks pushes that
modify `.github/workflows/*` without the `workflow` scope; the safer split
is shipping the Dockerfile + the action shell here, and adding the publish
workflow when the maintainer next pushes from a workstation.
