"""Fixed-budget v8 exploration: exact affected-facility drop feasibility.

The v7b encoder is unchanged, permitting a controlled old-checkpoint ablation.
Masks simulate inventory balance only; they never evaluate or rank objectives.
"""
from hazardous_waste_model import pickup_type
from src.ppo_objects_v7b import ObjectState as PreviousState


class ObjectState(PreviousState):
    def prepare(self, plan, solution=None):
        if getattr(self, '_strict_cached_plan', None) is plan:
            return
        super().prepare(plan, solution)
        self._processing_safe_drops = set()
        for slot in self._drops:
            node, period = self.pickups[slot]
            future, future_key, _, _, quantity = self._next[slot]
            source_key, _ = self._members[slot]
            changes = {}
            for key, t, delta in ((source_key, period, -quantity),
                                  (future_key, future, quantity)):
                address = (plan[key][0], pickup_type(node), t)
                changes[address] = changes.get(address, 0.) + delta
            if self.processing_feasible(changes):
                self._processing_safe_drops.add(slot)
        self._strict_cached_plan = plan

    def processing_feasible(self, changes):
        """Replay earliest feasible processing for changed receipts (not routes).

        With no lower processing bounds, maximum available processing minimizes
        each inventory. Exceeding storage or failing terminal clearance then
        cannot be repaired by a different processing schedule on these routes.
        """
        p = self.p
        affected = {(j, s) for j, s, t in changes}
        inventories = {(j, s): p.initial_facility_inventory[j, s] for j, s in affected}
        facilities = {j for j, s in affected}
        for t in p.periods:
            before = {}
            for j, s in affected:
                before[j, s] = (inventories[j, s] + self._raw.get(('R', j, s, t), 0.)
                                + changes.get((j, s, t), 0.))
                if before[j, s] < -1e-8:
                    return False
                capacity = p.processing_capacity[j, s, t] * p.technology[j, s]
                inventories[j, s] = max(0., before[j, s] - capacity)
            for j in facilities:
                total = sum(before[j, s] if (j, s) in affected
                            else self._raw.get(('BD', j, s, t), 0.) for s in p.waste_types)
                if total > p.facility_capacity[j] + 1e-8:
                    return False
        return all(value <= 1e-8 for value in inventories.values())

    def masks(self, plan, chosen=()):
        base = super().masks(plan, chosen)
        if not chosen:
            base[0] = bool(self._processing_safe_drops)
        elif chosen[0] == 0 and len(chosen) == 1:
            base &= self.mask(self._processing_safe_drops)
        return base
