use std::{
    fmt::{self, Debug, Display},
    marker::PhantomData,
    num::ParseIntError,
    str::FromStr,
    sync::atomic::{AtomicU64, Ordering},
};

use serde::{Deserialize, Serialize};
use sqlx::{Database, encode::IsNull};
use uuid::Uuid;

/// A generic identifier type that wraps a UUID and a type marker.
///
/// # Type Parameters:
///
/// * [`TypeMarker`]: A marker type used to differentiate between different types of IDs.
///
/// # Examples
///
/// ```rust
/// #[derive(Debug, PartialEq, Eq)]
/// enum SomeTypeIdMarker {}
/// type SomeTypeId = Id<SomeTypeIdMarker>;
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Id<TypeMarker: Debug + PartialEq + Eq>(Uuid, PhantomData<TypeMarker>);

impl<TypeMarker: Debug + PartialEq + Eq> Default for Id<TypeMarker> {
    fn default() -> Self {
        Self::new()
    }
}

impl<TypeMarker: Debug + PartialEq + Eq> Id<TypeMarker> {
    #[must_use]
    pub fn new() -> Self {
        Self(Uuid::new_v4(), PhantomData)
    }

    #[must_use]
    pub const fn from(uid: Uuid) -> Self {
        Self(uid, PhantomData)
    }

    #[must_use]
    pub const fn as_uuid_ref(&self) -> &Uuid {
        &self.0
    }

    #[must_use]
    pub const fn as_bytes(&self) -> &UuidBytes {
        self.0.as_bytes()
    }
}

impl<TypeMarker: Debug + PartialEq + Eq> Display for Id<TypeMarker> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        Display::fmt(&self.0, f)
    }
}

impl<TypeMarker: Debug + PartialEq + Eq> FromStr for Id<TypeMarker> {
    type Err = uuid::Error;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        Uuid::from_str(value).map(Self::from)
    }
}

impl<TypeMarker, Db> sqlx::Type<Db> for Id<TypeMarker>
where
    TypeMarker: Debug + PartialEq + Eq,
    Db: Database,
    Uuid: sqlx::Type<Db>,
{
    fn type_info() -> <Db as Database>::TypeInfo {
        <Uuid as sqlx::Type<Db>>::type_info()
    }

    fn compatible(ty: &<Db as Database>::TypeInfo) -> bool {
        <Uuid as sqlx::Type<Db>>::compatible(ty)
    }
}

impl<'encode, TypeMarker, Db> sqlx::Encode<'encode, Db> for Id<TypeMarker>
where
    TypeMarker: Debug + PartialEq + Eq,
    Db: Database,
    Uuid: sqlx::Encode<'encode, Db>,
{
    fn encode_by_ref(
        &self,
        buf: &mut <Db as Database>::ArgumentBuffer<'encode>,
    ) -> Result<IsNull, sqlx::error::BoxDynError> {
        self.0.encode_by_ref(buf)
    }
}

impl<'decode, TypeMarker, Db> sqlx::Decode<'decode, Db> for Id<TypeMarker>
where
    TypeMarker: Debug + PartialEq + Eq,
    Db: Database,
    Uuid: sqlx::Decode<'decode, Db>,
{
    fn decode(
        value: <Db as Database>::ValueRef<'decode>,
    ) -> Result<Self, sqlx::error::BoxDynError> {
        Uuid::decode(value).map(|uuid| Self(uuid, PhantomData))
    }
}

pub type UuidBytes = uuid::Bytes;

/// A generic identifier backed by a `u64`, intended for database `BIGINT UNSIGNED
/// AUTO_INCREMENT` columns.
///
/// Same shape as [`Id`], but stored as an 8-byte integer instead of a UUID. The type marker
/// makes ids of different kinds non-interchangeable at the type level.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct IntId<TypeMarker: Debug + PartialEq + Eq>(u64, PhantomData<TypeMarker>);

impl<TypeMarker: Debug + PartialEq + Eq> Default for IntId<TypeMarker> {
    fn default() -> Self {
        Self::new()
    }
}

static INT_ID_COUNTER: AtomicU64 = AtomicU64::new(1);

impl<TypeMarker: Debug + PartialEq + Eq> IntId<TypeMarker> {
    /// Returns a fresh in-process id. The DB-side `AUTO_INCREMENT` is the source of truth for
    /// persisted ids; this is only used for test fixtures or as a placeholder before insertion.
    #[must_use]
    pub fn new() -> Self {
        Self(INT_ID_COUNTER.fetch_add(1, Ordering::Relaxed), PhantomData)
    }

    #[must_use]
    pub const fn from(value: u64) -> Self {
        Self(value, PhantomData)
    }

    #[must_use]
    pub const fn as_u64(&self) -> u64 {
        self.0
    }
}

impl<TypeMarker: Debug + PartialEq + Eq> Display for IntId<TypeMarker> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        Display::fmt(&self.0, f)
    }
}

impl<TypeMarker: Debug + PartialEq + Eq> FromStr for IntId<TypeMarker> {
    type Err = ParseIntError;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        u64::from_str(value).map(Self::from)
    }
}

impl<TypeMarker, Db> sqlx::Type<Db> for IntId<TypeMarker>
where
    TypeMarker: Debug + PartialEq + Eq,
    Db: Database,
    u64: sqlx::Type<Db>,
{
    fn type_info() -> <Db as Database>::TypeInfo {
        <u64 as sqlx::Type<Db>>::type_info()
    }

    fn compatible(ty: &<Db as Database>::TypeInfo) -> bool {
        <u64 as sqlx::Type<Db>>::compatible(ty)
    }
}

impl<'encode, TypeMarker, Db> sqlx::Encode<'encode, Db> for IntId<TypeMarker>
where
    TypeMarker: Debug + PartialEq + Eq,
    Db: Database,
    u64: sqlx::Encode<'encode, Db>,
{
    fn encode_by_ref(
        &self,
        buf: &mut <Db as Database>::ArgumentBuffer<'encode>,
    ) -> Result<IsNull, sqlx::error::BoxDynError> {
        self.0.encode_by_ref(buf)
    }
}

impl<'decode, TypeMarker, Db> sqlx::Decode<'decode, Db> for IntId<TypeMarker>
where
    TypeMarker: Debug + PartialEq + Eq,
    Db: Database,
    u64: sqlx::Decode<'decode, Db>,
{
    fn decode(
        value: <Db as Database>::ValueRef<'decode>,
    ) -> Result<Self, sqlx::error::BoxDynError> {
        u64::decode(value).map(|v| Self(v, PhantomData))
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ResourceGroupIdMarker {}
pub type ResourceGroupId = Id<ResourceGroupIdMarker>;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TaskIdMarker {}
pub type TaskId = Id<TaskIdMarker>;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum JobIdMarker {}
pub type JobId = IntId<JobIdMarker>;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DataIdMarker {}
pub type DataId = Id<DataIdMarker>;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ExecutionManagerIdMarker {}
pub type ExecutionManagerId = IntId<ExecutionManagerIdMarker>;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SchedulerIdMarker {}
pub type SchedulerId = Id<SchedulerIdMarker>;

pub type SessionId = u64;

pub type TaskInstanceId = u64;

/// Represents a signed ID.
///
/// In the Spider scheduling framework, resources are owned by resource groups. Many operations
/// require both the resource group ID and the resource's own ID to enforce proper access control.
/// This struct encapsulates both identifiers for such operations by treating the resource group ID
/// as the signature.
///
/// # Type Parameters
///
/// * [`TypeMarker`] - A marker type used to differentiate between different resource types.
pub struct SignedId<IdType> {
    signature: ResourceGroupId,
    id: IdType,
}

impl<IdType> SignedId<IdType> {
    /// Factory function.
    ///
    /// # Returns
    ///
    /// A newly created instance of [`SignedId`].
    #[must_use]
    pub const fn new(signature: ResourceGroupId, id: IdType) -> Self {
        Self { signature, id }
    }

    /// # Returns
    ///
    /// A reference to the underlying signature.
    #[must_use]
    pub const fn get_signature(&self) -> &ResourceGroupId {
        &self.signature
    }

    /// # Returns
    ///
    /// A reference to the underlying raw ID.
    #[must_use]
    pub const fn get(&self) -> &IdType {
        &self.id
    }
}

pub type SignedJobId = SignedId<JobId>;

pub type SignedTaskId = SignedId<TaskId>;

#[cfg(test)]
mod tests {
    use std::any::TypeId;

    use super::*;

    #[test]
    fn test_id_basic() {
        let id = TaskId::new();
        let underlying_uuid = id.as_uuid_ref().to_owned();
        assert_eq!(id, TaskId::from(underlying_uuid));

        assert_ne!(TypeId::of::<TaskId>(), TypeId::of::<JobId>());
    }

    #[test]
    fn task_id_json_roundtrip() {
        let id = TaskId::new();
        let deserialized_id: TaskId = serde_json::from_str(
            serde_json::to_string(&id)
                .expect("JSON serialization failure")
                .as_str(),
        )
        .expect("JSON deserialization failure");
        assert_eq!(id, deserialized_id);
    }

    #[test]
    fn int_id_format_and_parse() {
        let id = JobId::from(42);
        assert_eq!(id.to_string(), "42");
        let parsed: JobId = "42".parse().expect("parse should succeed");
        assert_eq!(parsed, id);
    }

    #[test]
    fn int_id_new_is_monotonic() {
        let a = JobId::new();
        let b = JobId::new();
        assert_ne!(a, b);
    }
}
