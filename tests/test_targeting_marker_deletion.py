import unittest
from types import SimpleNamespace
from unittest.mock import Mock

import invesalius.constants as const
from invesalius.gui.task_navigator import MarkersPanel


class MarkerDeletionTests(unittest.TestCase):
    def state(self, rows):
        # Each row holds UUID, model ID, and sort-map key in visible order.
        rows = [list(row) for row in rows]
        table = Mock()
        table.GetItemCount.side_effect = lambda: len(rows)
        table.GetItem.side_effect = lambda row, column: SimpleNamespace(GetText=lambda: str(rows[row][0 if column == const.UUID else 1]))
        table.GetItemData.side_effect = lambda row: rows[row][2]
        table.DeleteItem.side_effect = lambda row: rows.pop(row)
        table.SetItem.side_effect = lambda row, column, text: rows[row].__setitem__(1, int(text))
        state = SimpleNamespace(marker_list_ctrl=table, itemDataMap={})
        state._MarkersPanel__get_marker_id = lambda row: rows[row][1]
        state._MarkersPanel__find_marker_index = lambda marker_id: next((i for i, row in enumerate(rows) if row[1] == marker_id), None)
        for uuid, marker_id, key in rows:
            data = [''] * (const.UUID + 2)
            data[const.ID_COLUMN] = marker_id
            data[-1] = uuid
            state.itemDataMap[key] = data
        return state, rows

    def test_deletes_correct_uuid_with_sorted_rows_and_updates_ids(self):
        state, rows = self.state([('last', 2, 30), ('deleted', 1, 20), ('first', 0, 10)])
        MarkersPanel._DeleteMarker(state, SimpleNamespace(marker_id=1, marker_uuid='deleted'))
        self.assertEqual(rows, [['last', 1, 30], ['first', 0, 10]])
        self.assertNotIn(20, state.itemDataMap)
        self.assertEqual(state.itemDataMap[30][const.ID_COLUMN], '1')

    def test_absent_row_does_not_delete_another_marker_and_cleans_stale_metadata(self):
        state, rows = self.state([('other', 1, 20)])
        state.itemDataMap[99] = ['stale', 'deleted']
        MarkersPanel._DeleteMarker(state, SimpleNamespace(marker_id=1, marker_uuid='deleted'))
        state.marker_list_ctrl.DeleteItem.assert_not_called()
        self.assertEqual(rows, [['other', 1, 20]])
        self.assertNotIn(99, state.itemDataMap)

    def test_removes_all_stale_metadata_without_mutating_dictionary_during_iteration(self):
        state, rows = self.state([])
        state.itemDataMap = {1: ['deleted'], 2: ['deleted'], 3: ['other']}
        MarkersPanel._DeleteMarker(state, SimpleNamespace(marker_id=0, marker_uuid='deleted'))
        self.assertEqual(state.itemDataMap, {3: ['other']})

    def test_legacy_marker_without_uuid_uses_id_and_row_key(self):
        state, rows = self.state([('', 0, 10), ('other', 1, 20)])
        MarkersPanel._DeleteMarker(state, SimpleNamespace(marker_id=0, marker_uuid=None))
        self.assertEqual(rows, [['other', 0, 20]])
        self.assertNotIn(10, state.itemDataMap)
