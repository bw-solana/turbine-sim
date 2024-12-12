use {
    hdrhistogram::Histogram,
    priority_queue::PriorityQueue,
    rand::Rng,
    std::{
        cmp::Reverse,
        collections::HashMap,
        ops::Range,
        sync::{Arc, Mutex},
        thread,
        time::{Duration, Instant},
    },
};

// Number of blocks to simulate
const NUM_ITERATIONS: usize = 10;
// Elongates time to make the non-network parts of the simulation seem
// infinitely fast. Larger value here makes the sim take longer. Shorter value
// may introduce non-network delays into the results.
const TIME_FACTOR: u64 = 30;
// Putting some reasonable upper bound just to help size vectors
const MAX_SHREDS: usize = 10_000;
// Assume 32:32 encoding
const SHREDS_PER_ERASURE_BATCH: usize = 64;
const MAX_ERASURE_BATCHES: usize = MAX_SHREDS.div_ceil(SHREDS_PER_ERASURE_BATCH);
// This value was chosen empirically from looking at
// broadcast-transmit-shreds-stats.end_to_end_elapsed metrics
const TARGET_SLOT_TIME_MS: usize = 340;
// This value was chosen empirically from looking at
// broadcast-transmit-shreds-stats.num_shreds metrics
const NUM_SHREDS: usize = 1_550;
// This is the max number of nodes the leader will transmit a shred to.
const LEADER_FANOUT: usize = 1;
// This is the max number of nodes retransmitters will retransmit a shred to.
const FANOUT: usize = 200;
// Leader, Root, L1, L2
const MAX_TURBINE_LAYERS: usize = 4;
// Approximate number of staked nodes on mainnet.
const NUM_NODES: usize = 1500;
const PACKET_DROP_PCT: usize = 1;

struct BlockDeliveryMetrics {
    first_shred_received_time: Instant,
    last_shred_received_time: Instant,
    final_shred_layer: usize,
    final_shred_layer_times: Vec<Instant>,
    was_recovered: bool,
}

#[derive(Clone, Hash, PartialEq, Eq)]
struct Shred {
    index: usize,
    arrival_times: Vec<Option<Instant>>,
    receiving_layer: usize,
    was_recovered: bool,
}

/// Node in the network
#[derive(Clone)]
struct Node {
    shred_queue: Arc<Mutex<PriorityQueue<Shred, Reverse<Instant>>>>,
    blockstore: Blockstore,
}

#[derive(Clone)]
struct ErasureBatch {
    num_shreds_received: usize,
    received: u64,
}

#[derive(Clone)]
struct Blockstore {
    num_shreds_received: usize,
    erasure_batch: Vec<ErasureBatch>,
}

/// Simulation Parameters
#[derive(Clone)]
struct SimulationParams {
    num_nodes: usize,
    leader_fanout: usize,
    fanout: usize,
    num_shreds: usize,
    stake_range: (usize, usize), // Min and max stake
    network_delay: Range<u64>,   // Min and max delay in milliseconds
}

struct SplitTimes {
    node_index: usize,
    turbine_layer: usize,
    was_recovered: bool,
    // All times in ms
    shred_creation: u64,
    layer_0: u64,
    layer_1: u64,
    layer_2: u64,
}
struct SimulationStats {
    histogram_start_produce: Histogram<u64>,
    histogram_first_rcv: Histogram<u64>,
    final_shred_layer_0_count: usize,
    final_shred_layer_1_count: usize,
    final_shred_layer_2_count: usize,
    final_shred_split_times: Vec<SplitTimes>,
}

impl Default for SimulationStats {
    fn default() -> Self {
        SimulationStats {
            histogram_start_produce: Histogram::<u64>::new(3).unwrap(),
            histogram_first_rcv: Histogram::<u64>::new(3).unwrap(),
            final_shred_layer_0_count: 0,
            final_shred_layer_1_count: 0,
            final_shred_layer_2_count: 0,
            final_shred_split_times: Vec::with_capacity(NUM_NODES),
        }
    }
}

impl SimulationStats {
    fn collect_stats(
        delivery_times: Arc<Mutex<HashMap<usize, BlockDeliveryMetrics>>>,
        start_time: Instant,
    ) -> SimulationStats {
        let mut stats = SimulationStats::default();
        let delivery_times = delivery_times.lock().unwrap();
        for (
            node_id,
            BlockDeliveryMetrics {
                first_shred_received_time,
                last_shred_received_time,
                final_shred_layer,
                final_shred_layer_times,
                was_recovered,
            },
        ) in delivery_times.iter()
        {
            let leader_send_time = if *was_recovered {
                0
            } else {
                final_shred_layer_times[0]
                    .duration_since(start_time)
                    .as_millis() as u64
                    / TIME_FACTOR
            };
            let root_arrival_time = final_shred_layer_times[1]
                .duration_since(final_shred_layer_times[0])
                .as_millis() as u64
                / TIME_FACTOR;
            let l1_arrival_time = final_shred_layer_times[2]
                .duration_since(final_shred_layer_times[1])
                .as_millis() as u64
                / TIME_FACTOR;
            let l2_arrival_time = final_shred_layer_times[3]
                .duration_since(final_shred_layer_times[2])
                .as_millis() as u64
                / TIME_FACTOR;
            stats.final_shred_split_times.push(SplitTimes {
                node_index: *node_id,
                turbine_layer: *final_shred_layer,
                was_recovered: *was_recovered,
                shred_creation: leader_send_time,
                layer_0: root_arrival_time,
                layer_1: l1_arrival_time,
                layer_2: l2_arrival_time,
            });
            
            stats
                .histogram_start_produce
                .record(
                    last_shred_received_time
                        .duration_since(start_time)
                        .as_millis() as u64,
                )
                .unwrap();
            stats
                .histogram_first_rcv
                .record(
                    last_shred_received_time
                        .duration_since(*first_shred_received_time)
                        .as_millis() as u64,
                )
                .unwrap();
            match final_shred_layer {
                0 => stats.final_shred_layer_0_count += 1,
                1 => stats.final_shred_layer_1_count += 1,
                2 => stats.final_shred_layer_2_count += 1,
                _ => panic!("Unexpected turbine layer"),
            }
        }
        stats
    }

    fn output_final_shred_stats(&self) {
        for split_time in &self.final_shred_split_times {
            if split_time.turbine_layer < 2 {
                println!(
                    "Node {}: layer={}, recovered={} - {:?} / {:?} / {:?} ",
                    split_time.node_index,
                    split_time.turbine_layer,
                    split_time.was_recovered,
                    split_time.shred_creation,
                    split_time.layer_0,
                    split_time.layer_1
                );
            } else {
                println!(
                    "Node {}: layer={}, recovered={} - {:?} / {:?} / {:?} / {:?} ",
                    split_time.node_index,
                    split_time.turbine_layer,
                    split_time.was_recovered,
                    split_time.shred_creation,
                    split_time.layer_0,
                    split_time.layer_1,
                    split_time.layer_2
                );
            }
        }
        println!("\n  ### Final Shred Stats ###");
        println!("   Final shred distribution:");
        println!("     Layer 0: {} nodes", self.final_shred_layer_0_count);
        println!("     Layer 1: {} nodes", self.final_shred_layer_1_count);
        println!("     Layer 2: {} nodes", self.final_shred_layer_2_count);
    }

    fn output_stats(&self) {
        self.output_final_shred_stats();
        println!("\n  ### Simulation Results ###");
        println!("   Time from start produce:");
        println!(
            "     Min: {}ms",
            self.histogram_start_produce.min() / TIME_FACTOR
        );
        println!(
            "     P50: {}ms",
            self.histogram_start_produce.value_at_quantile(0.5) / TIME_FACTOR
        );
        println!(
            "     AVG: {}ms",
            self.histogram_start_produce.mean().round() / TIME_FACTOR as f64
        );
        println!(
            "     Max: {}ms",
            self.histogram_start_produce.max() / TIME_FACTOR
        );
        println!("   Time from first receive:");
        println!(
            "     Min: {}ms",
            self.histogram_first_rcv.min() / TIME_FACTOR
        );
        println!(
            "     P50: {}ms",
            self.histogram_first_rcv.value_at_quantile(0.5) / TIME_FACTOR
        );
        println!(
            "     AVG: {}ms",
            self.histogram_first_rcv.mean().round() / TIME_FACTOR as f64
        );
        println!(
            "     Max: {}ms",
            self.histogram_first_rcv.max() / TIME_FACTOR
        );
    }

    /// Output turbine latency statistics
    fn collect_and_output_stats(
        delivery_times: Arc<Mutex<HashMap<usize, BlockDeliveryMetrics>>>,
        start_time: Instant,
    ) {
        let stats = Self::collect_stats(delivery_times, start_time);
        Self::output_stats(&stats);
    }
}

/// Simulation State
struct Simulation {
    // Node index --> Node (shred queue & blockstore)
    nodes: Vec<Node>,
    // Pre-computed turbine trees for each shred
    turbine_trees: Vec<Vec<usize>>,
    params: SimulationParams,
    // Time at which the simulation started
    start_time: Instant,
    // Time at which each node received all shreds
    delivery_times: Arc<Mutex<HashMap<usize, BlockDeliveryMetrics>>>,
}

fn weighted_shuffle(items: &[(usize, usize)]) -> Vec<usize> {
    // Validate input
    assert!(!items.is_empty());

    // Create cumulative weights
    let mut cumulative_weights = Vec::new();
    let mut total_weight = 0;

    for &(_, weight) in items {
        total_weight += weight;
        cumulative_weights.push(total_weight);
    }

    // Generate the shuffled result
    let mut rng = rand::thread_rng();
    let mut result = Vec::new();
    let mut indices: Vec<usize> = (0..items.len()).collect();

    while !indices.is_empty() {
        // Draw a random value between 0 and total_weight
        let random_weight = rng.gen_range(0..total_weight);

        // Binary search to find the index corresponding to the random weight
        let index = cumulative_weights
            .binary_search_by(|&w| w.partial_cmp(&random_weight).unwrap())
            .unwrap_or_else(|i| i);

        // Add the item to the result
        let selected_index = indices.remove(index);
        result.push(items[selected_index].0);

        // Update cumulative weights and total weight
        let weight_to_remove = items[selected_index].1;
        total_weight -= weight_to_remove;

        for i in index..cumulative_weights.len() - 1 {
            cumulative_weights[i] = cumulative_weights[i + 1] - weight_to_remove;
        }
        cumulative_weights.pop();
    }

    result
}

#[inline]
fn get_turbine_children(
    turbine_tree: &[usize],
    node_index: usize,
    leader_fanout: usize,
    fanout: usize,
) -> Vec<usize> {
    let turbine_tree_index = turbine_tree.iter().position(|&x| x == node_index).unwrap();
    let (step, mut children) = if turbine_tree_index < leader_fanout {
        // Layer 0
        (leader_fanout, Vec::with_capacity(leader_fanout))
    } else if turbine_tree_index < leader_fanout + fanout.saturating_mul(leader_fanout) {
        // Layer 1
        (
            leader_fanout.saturating_mul(fanout),
            Vec::with_capacity(fanout),
        )
    } else {
        // Layer 2
        (
            leader_fanout.saturating_mul(fanout).saturating_mul(fanout),
            Vec::new(),
        )
    };

    for i in 1..=fanout {
        let index = turbine_tree_index + i.saturating_mul(step);
        if index < turbine_tree.len() {
            children.push(turbine_tree[index]);
        } else {
            break;
        }
    }

    children
}

fn get_turbine_layer(
    turbine_tree: &[usize],
    node_index: usize,
    leader_fanout: usize,
    fanout: usize,
) -> usize {
    let turbine_tree_index = turbine_tree.iter().position(|&x| x == node_index).unwrap();
    if turbine_tree_index < leader_fanout {
        // Layer 0
        0
    } else if turbine_tree_index < leader_fanout + fanout.saturating_mul(leader_fanout) {
        // Layer 1
        1
    } else {
        // Layer 2
        2
    }
}

fn drop_packet() -> bool {
    let mut rng = rand::thread_rng();
    rng.gen_range(0..100) <= PACKET_DROP_PCT // Generates a number between 0 and 99, returns true for 0
}

impl Simulation {
    /// Initialize the simulation with nodes and fanout
    fn new(params: SimulationParams) -> Self {
        // Build cluster of nodes
        let mut nodes = Vec::new();
        let mut node_stake = Vec::new();
        for id in 0..params.num_nodes {
            let stake =
                rand::thread_rng().gen_range(params.stake_range.0..=params.stake_range.1) as usize;
            let shred_queue = Arc::new(Mutex::new(PriorityQueue::new()));
            let blockstore = Blockstore {
                num_shreds_received: 0,
                erasure_batch: vec![
                    ErasureBatch {
                        num_shreds_received: 0,
                        received: 0
                    };
                    MAX_ERASURE_BATCHES
                ],
            };
            nodes.push(Node {
                shred_queue,
                blockstore,
            });
            node_stake.push((id, stake));
        }

        let mut turbine_trees = Vec::new();
        for _ in 0..params.num_shreds {
            let turbine_tree = weighted_shuffle(&node_stake);
            turbine_trees.push(turbine_tree);
        }

        Simulation {
            nodes,
            turbine_trees,
            params,
            start_time: Instant::now(),
            delivery_times: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    #[inline]
    fn retransmit_shred(
        cluster_nodes: &[Node],
        shred: &mut Shred,
        children: &Vec<usize>,
        network_delay: Range<u64>,
    ) {
        if drop_packet() {
            return;
        }
        shred.receiving_layer += 1;
        for child_id in children {
            let child_node = &cluster_nodes[*child_id];
            let mut shred_queue = child_node.shred_queue.lock().unwrap();
            let send_time = Instant::now()
                + Duration::from_millis(
                    TIME_FACTOR * rand::thread_rng().gen_range(network_delay.clone()),
                );
            shred.arrival_times[shred.receiving_layer] = Some(send_time);

            // If this shred is already in the queue, only keep the earliest version
            if let Some(existing_time) = shred_queue.get_priority(shred) {
                if send_time < existing_time.0 {
                    shred_queue.change_priority(shred, Reverse(send_time));
                }
            } else {
                shred_queue.push(shred.clone(), Reverse(send_time));
            }
        }
    }

    #[inline]
    fn retransmit_recovered_shred(
        cluster_nodes: &[Node],
        shred: &mut Shred,
        children: &Vec<usize>,
        network_delay: Range<u64>,
        turbine_layer: usize,
    ) {
        shred.receiving_layer = turbine_layer;
        shred.arrival_times[shred.receiving_layer] = Some(Instant::now());
        Self::retransmit_shred(cluster_nodes, shred, children, network_delay);
    }

    // Returns true if shred is duplicate
    #[inline]
    fn insert_shred(shred_index: usize, erasure_batch: usize, blockstore: &mut Blockstore) -> bool {
        let erasure_data = &mut blockstore.erasure_batch[erasure_batch];

        // Early return if the erasure batch is already complete
        if erasure_data.num_shreds_received == SHREDS_PER_ERASURE_BATCH {
            return true;
        }

        // Update erasure batch data
        erasure_data.received |= 1 << (shred_index % SHREDS_PER_ERASURE_BATCH);
        erasure_data.num_shreds_received += 1;

        // Update the total shreds received
        blockstore.num_shreds_received += 1;

        false
    }

    // Returns recovered shred indices
    #[inline]
    fn attempt_erasure_recovery(
        erasure_batch: usize,
        blockstore: &mut Blockstore,
        recovered_shreds: &mut Vec<Shred>,
    ) {
        let erasure_data = &mut blockstore.erasure_batch[erasure_batch];

        // Check if enough shreds have been received to perform recovery
        if erasure_data.num_shreds_received < SHREDS_PER_ERASURE_BATCH / 2 {
            return;
        }

        // Iterate over all possible shred indices in the erasure batch
        let base_shred_index = erasure_batch * SHREDS_PER_ERASURE_BATCH;
        let missing_mask = !erasure_data.received;
        for shred_index in 0..SHREDS_PER_ERASURE_BATCH {
            if missing_mask & (1 << shred_index) != 0 {
                // Recover the missing shred
                let recovered_shred = Shred {
                    index: base_shred_index + shred_index,
                    arrival_times: vec![None; MAX_TURBINE_LAYERS],
                    receiving_layer: 0,
                    was_recovered: true,
                };
                recovered_shreds.push(recovered_shred);
            }
        }

        // Update blockstore with the recovered shreds
        let recovered_count = SHREDS_PER_ERASURE_BATCH - erasure_data.num_shreds_received;
        erasure_data.received = u64::MAX;
        erasure_data.num_shreds_received = SHREDS_PER_ERASURE_BATCH;
        blockstore.num_shreds_received += recovered_count;
    }

    fn retransmit_recovered_shreds(
        cluster_nodes: &[Node],
        recovered_shreds: &mut Vec<Shred>,
        network_delay: Range<u64>,
        shred_to_children: &[Vec<usize>],
        turbine_trees: &[Vec<usize>],
        node_id: usize,
    ) {
        for recovered_shred in recovered_shreds {
            let recovered_shred_index = recovered_shred.index;
            Self::retransmit_recovered_shred(
                cluster_nodes,
                recovered_shred,
                &shred_to_children[recovered_shred_index],
                network_delay.clone(),
                get_turbine_layer(
                    &turbine_trees[recovered_shred_index],
                    node_id,
                    LEADER_FANOUT,
                    FANOUT,
                ),
            );
        }
    }

    // Returns true if all shreds have been received
    #[inline]
    fn record_shred_timing(
        node_id: usize,
        num_shreds_received: usize,
        num_shreds: usize,
        delivery_times: &Mutex<HashMap<usize, BlockDeliveryMetrics>>,
        turbine_layer: usize,
        shred: &Shred,
    ) -> bool {
        if num_shreds_received == 1 {
            // First shred was received
            let block_delivery_metrics = BlockDeliveryMetrics {
                first_shred_received_time: Instant::now(),
                last_shred_received_time: Instant::now(),
                final_shred_layer: turbine_layer,
                final_shred_layer_times: vec![
                    Instant::now() + Duration::from_secs(1000);
                    MAX_TURBINE_LAYERS
                ],
                was_recovered: shred.was_recovered,
            };
            delivery_times
                .lock()
                .unwrap()
                .insert(node_id, block_delivery_metrics);
        } else if num_shreds_received == num_shreds {
            // Last shred was received
            let mut delivery_times = delivery_times.lock().unwrap();
            let block_delivery_metrics = delivery_times.get_mut(&node_id).unwrap();
            for layer in 0..MAX_TURBINE_LAYERS {
                let Some(time) = shred.arrival_times[layer] else {
                    continue;
                };
                block_delivery_metrics.final_shred_layer_times[layer] = time;
            }
            block_delivery_metrics.final_shred_layer_times[turbine_layer + 1] = Instant::now();
            block_delivery_metrics.last_shred_received_time = Instant::now();
            block_delivery_metrics.final_shred_layer = turbine_layer;
            block_delivery_metrics.was_recovered = shred.was_recovered;
            return true;
        }

        false
    }

    #[inline]
    fn retransmit_thread(
        nodes: &mut [(Node, usize, bool, Vec<Vec<usize>>)],
        cluster_nodes: &[Node],
        delivery_times: &Mutex<HashMap<usize, BlockDeliveryMetrics>>,
        network_delay: Range<u64>,
        num_shreds: usize,
        turbine_trees: &[Vec<usize>],
    ) {
        let mut num_nodes_finished = 0;
        let num_nodes = nodes.len();
        let mut local_ready_shreds = Vec::with_capacity(MAX_SHREDS);
        let mut recovered_shreds = Vec::with_capacity(SHREDS_PER_ERASURE_BATCH / 2);
        loop {
            for (node, node_id, finished, shred_to_children) in nodes.iter_mut() {
                if *finished {
                    // This validator has already finished
                    continue;
                }

                {
                    // Batch shred processing to reduce lock contention
                    if let Ok(mut shred_queue) = node.shred_queue.try_lock() {
                        while let Some((_, Reverse(ready_time))) = shred_queue.peek() {
                            let time_now = Instant::now();
                            if *ready_time <= time_now {
                                if time_now - *ready_time > Duration::from_millis(100) {
                                    println!(
                                        "!!! WARN: Node {} is behind in processing by {}ms - Increase TIME_FACTOR !!!",
                                        node_id,
                                        (time_now - *ready_time).as_millis()
                                    );
                                }
                                local_ready_shreds.push(shred_queue.pop().unwrap().0);
                            } else {
                                break;
                            }
                        }
                    }
                }

                for shred in &mut local_ready_shreds {
                    let shred_index = shred.index;
                    let erasure_batch = shred_index / SHREDS_PER_ERASURE_BATCH;
                    let blockstore = &mut node.blockstore;

                    if Self::insert_shred(shred_index, erasure_batch, blockstore) {
                        // Duplicate shred
                        continue;
                    }

                    Self::retransmit_shred(
                        cluster_nodes,
                        shred,
                        &shred_to_children[shred_index],
                        network_delay.clone(),
                    );

                    Self::attempt_erasure_recovery(
                        erasure_batch,
                        blockstore,
                        &mut recovered_shreds,
                    );

                    Self::retransmit_recovered_shreds(
                        cluster_nodes,
                        &mut recovered_shreds,
                        network_delay.clone(),
                        shred_to_children,
                        turbine_trees,
                        *node_id,
                    );
                    recovered_shreds.clear();

                    if Self::record_shred_timing(
                        *node_id,
                        blockstore.num_shreds_received,
                        num_shreds,
                        delivery_times,
                        get_turbine_layer(
                            &turbine_trees[shred_index],
                            *node_id,
                            LEADER_FANOUT,
                            FANOUT,
                        ),
                        shred,
                    ) {
                        *finished = true;
                        num_nodes_finished += 1;
                        if num_nodes_finished >= num_nodes {
                            // All nodes have completed
                            return;
                        }
                    }
                }

                local_ready_shreds.clear();
            }
        }
    }

    fn deploy_retransmit_threads(&self) -> Vec<thread::JoinHandle<()>> {
        const VALIDATORS_PER_THREAD: usize = 30;
        let mut handles = Vec::new();
        let cluster_nodes = &self.nodes.clone();
        let num_shreds = self.params.num_shreds;

        for start_node_id in (0..cluster_nodes.len()).step_by(VALIDATORS_PER_THREAD) {
            // Create a thread for a subset of nodes
            let mut nodes = Vec::with_capacity(VALIDATORS_PER_THREAD);
            for (node_id, node) in cluster_nodes
                .iter()
                .enumerate()
                .skip(start_node_id)
                .take(VALIDATORS_PER_THREAD)
            {
                // Precompute children in the turbine for each shred
                let mut shred_to_children = Vec::with_capacity(num_shreds);
                for shred in 0..num_shreds {
                    shred_to_children.push(get_turbine_children(
                        &self.turbine_trees[shred],
                        node_id,
                        self.params.leader_fanout,
                        self.params.fanout,
                    ));
                }
                nodes.push((node.clone(), node_id, false, shred_to_children));
            }
            let cluster_nodes = cluster_nodes.clone();
            let network_delay = self.params.network_delay.clone();
            let delivery_times = self.delivery_times.clone();
            let turbine_trees = self.turbine_trees.clone();
            let handle = thread::spawn(move || {
                Self::retransmit_thread(
                    &mut nodes,
                    &cluster_nodes,
                    &delivery_times,
                    network_delay,
                    num_shreds,
                    &turbine_trees,
                )
            });

            handles.push(handle);
        }

        handles
    }

    fn transmit_erasure_batch(&mut self, starting_shred_index: usize) {
        for index_in_erasure_batch in 0..SHREDS_PER_ERASURE_BATCH {
            let shred_index = starting_shred_index + index_in_erasure_batch;
            if shred_index >= self.params.num_shreds {
                break;
            }

            // Send to all root nodes
            for turbine_tree_id in 0..self.params.leader_fanout {
                let node_id = self.turbine_trees[shred_index][turbine_tree_id];
                let node = &self.nodes[node_id];
                let mut shred_queue = node.shred_queue.lock().unwrap();
                let arrival_time = Instant::now()
                    + Duration::from_millis(
                        TIME_FACTOR
                            * rand::thread_rng().gen_range(self.params.network_delay.clone()),
                    );
                let mut arrival_times = vec![None; MAX_TURBINE_LAYERS];
                arrival_times[0] = Some(Instant::now());
                arrival_times[1] = Some(arrival_time);
                let shred = Shred {
                    index: shred_index,
                    arrival_times,
                    receiving_layer: 1,
                    was_recovered: false,
                };
                assert_eq!(shred_queue.push(shred, Reverse(arrival_time)), None);
            }
        }
    }

    // Simulate leader sending shreds to all roots
    fn transmit_leader_block(&mut self) {
        let num_shreds = self.params.num_shreds;
        println!(
            "Starting to produce leader block with {} shreds",
            num_shreds
        );
        self.start_time = Instant::now();
        for starting_shred_index in (0..num_shreds).step_by(SHREDS_PER_ERASURE_BATCH) {
            self.transmit_erasure_batch(starting_shred_index);
            // Have the leader sleep to simulate the normal delays between erasure batches.
            let next_batch_send_time = self.start_time
                + Duration::from_millis(
                    TIME_FACTOR
                        * (TARGET_SLOT_TIME_MS * (starting_shred_index + SHREDS_PER_ERASURE_BATCH)
                            / num_shreds) as u64,
                );
            while Instant::now() < next_batch_send_time {
                thread::sleep(Duration::from_millis(1));
            }
        }
        println!(
            "Leader sent all shreds in {}ms",
            self.start_time.elapsed().as_millis() / TIME_FACTOR as u128
        );
    }

    /// Simulate the block delivery using Turbine
    fn simulate(&mut self) {
        println!("Starting simulation with {} nodes", self.params.num_nodes);
        let handles = self.deploy_retransmit_threads();
        self.transmit_leader_block();

        // Wait for all threads to finish
        for handle in handles {
            handle.join().unwrap();
        }
    }
}

fn main() {
    for iteration in 0..NUM_ITERATIONS {
        println!("\n### Iteration {} ###", iteration);
        let params = SimulationParams {
            num_nodes: NUM_NODES,
            leader_fanout: LEADER_FANOUT,
            fanout: FANOUT,
            num_shreds: NUM_SHREDS
                .checked_next_multiple_of(SHREDS_PER_ERASURE_BATCH)
                .unwrap(),
            stake_range: (1, 100),
            network_delay: 25..100,
        };

        let mut simulation = Simulation::new(params);
        simulation.simulate();
        SimulationStats::collect_and_output_stats(
            simulation.delivery_times.clone(),
            simulation.start_time,
        );
    }
}
