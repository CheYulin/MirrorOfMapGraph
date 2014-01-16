/*
 Copyright (C) SYSTAP, LLC 2006-2014.  All rights reserved.

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

 http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
 */

/*! \mainpage MPGraph

\section intro_sec Introduction

MPGraph is Massively Parallel Graph processing on GPUs.

The MPGraph API makes it easy to develop high performance graph
analytics on GPUs. The API is based on the Gather-Apply-Scatter (GAS)
model as used in GraphLab. To deliver high performance computation and
efficiently utilize the high memory bandwidth of GPUs, MPGraph's CUDA
kernels use multiple sophisticated strategies, such as
vertex-degree-dependent dynamic parallelism granularity and frontier
compaction.

MPGraph is up to two order of magnitude faster than parallel CPU
implementations on up 24 CPU cores and has performance comparable to a
state-of-the-art manually optimized GPU implementation.

New algorithms can be implemented in a few hours that fully exploit
the data-level parallelism of the GPU and offer throughput of up to 3
billion traversed edges per second on a single GPU.

Partitioned graphs and Multi-GPU support will be in a future release.

This work was (partially) funded by the DARPA XDATA program under AFRL
Contract #FA8750-13-C-0002.

\section api The MPGraph API

MPGraph is implemented as a set of templates following a design
pattern that is similar to the Gather-Apply-Scatter (GAS) API. GAS is
a vertex-centric API, similar to the API first popularized by Pregel.
The GAS API breaks down operations into the following phases:

- Gather  - reads data from the one-hop neighborhood of a vertex.
- Apply   - updates the vertex state based on the gather result.
- Scatter - pushes updates to the one-hop neighborhood of a vertex.

The GAS API has been extended in order to: (a) maximize parallelism;
(b) manage simultaneous discovery of duplicate vertices (this is not
an issue in multi-core CPU code); (c) provide appropriate memory
barriers (each kernel provides a memory barrier); (d) optimize memory
layout; and (e) allow "push" style scatter operators are similar to
"signal" with a message value to create a side-effect in GraphLab.

\subsection kernels MPGraph Kernels

MPGraph defines the following kernels and supports their invocation
from templated CUDA programs.  Each kernel may have one or more device
functions that it invokes.  User code (a) provides implementations of
those device functions to customize the behavior of the algorithm; and
(b) provides custom data structures for the vertices and links (see
below).

Gather Phase Kernels::

- gather: The gather kernel reads data from the one-hop neighborhood
  of each vertex in the frontier.

Apply Phase Kernels::

- apply: The apply kernel updates the state of each vertex in the
  frontier given the results of the most recent gather or scatter
  operation.

- post-apply: The post-apply kernel runs after all threads in the
  apply() function have had the opportunity to synchronize at a memory
  barrier.

Scatter Phase Kernels::

- expand: The expand kernel creates the new frontier.

- contract: The contract kernel eliminates duplicates in the frontier
  (this is the problem of simultaneous discovery).

\subsection data_structures MPGraph Data Structures

In order to write code to the MPGraph API, you need to be aware of the
following data structures:

- Frontier: The frontier is a dynamic queue containing those vertices
  that are active in each iteration.  The frontier is managed by the
  MPGraph kernels, but user data may be allocated and accessed that is
  1:1 with the frontier.  For example, BFS uses a scratch array to
  store the predecessor value.

- Topology: The topology is built from the sparse matrix data file. It
  is currently maintained in a CSR (Compressed Sparse Row) and/or CSC
  (Compressed Sparse Column) data structures, depending on the access
  patterns required by the Gather and/or Scatter phases.  Users do not
  have direct access to these data structures and the implementations
  are likely to evolve substantially, e.g., to support topology
  compression and decomposition of large graphs and MPGraph algorithms
  on GPU workstations and GPU clusters.

- Vertex list: The vertex list is a structure of arrays pattern.  See
  a VertexData structure in one of the existing algorithms for
  examples.

- Edge list: The edge list is a structure of arrays pattern. See an
  EdgeData structure in one of the existing algorithms for examples.

The user data (vertex list and edge list) are laid out in vertical
stripes using a Structures of Arrays pattern for optimal memory access
patterns on the GPU.  To add your own data, you add a field to the
vertex data struct or the edge data struct. That field will be an
array that is 1:1 with the vertex identifiers.  You will need to
initialize your array.  MPGraph will provide you with access to your
data from within the appropriate device functions.


\subsection write_your_own Writing your own MPGraph algorithm

MPGraph is based on templates.  This means that there is no interface
or super class from which you can derive your code.  Instead, you need
to start with one of the existing implementations that uses the
MPGraph template "pattern". You then need to review and modify the
function that initializes the user data structures (the vertex list
and the edge list) and the device functions that implement the user
code for the Gather, Apply, and Scatter primitives.  You can also
define functions that will extract the results from the GPU.

*/
